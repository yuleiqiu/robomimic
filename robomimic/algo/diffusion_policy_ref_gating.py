"""
Diffusion Policy variant with an external reference-image gating branch.
"""

import os
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.base_nets as BaseNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.models.obs_core as ObsCore
import robomimic.models.obs_nets as ObsNets
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet, replace_bn_with_gn
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict


def _load_reference_arrays(reference_bank_path):
    path = Path(os.path.expanduser(reference_bank_path))
    if not path.exists():
        raise FileNotFoundError("Reference bank path {} does not exist".format(path))

    if path.is_dir():
        image_paths = []
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(sorted(path.glob(suffix)))
        if len(image_paths) == 0:
            raise ValueError("Reference directory {} does not contain supported image files".format(path))
        return [imageio.imread(image_path) for image_path in image_paths]

    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if "images" in data:
                data = data["images"]
            else:
                data = next(iter(data.values()))
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
    elif suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        data_file = np.load(path)
        data = data_file["images"] if "images" in data_file else data_file[data_file.files[0]]
    else:
        data = imageio.imread(path)

    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return [data]
        if data.ndim == 4:
            return [data[i] for i in range(data.shape[0])]

    raise ValueError(
        "Unsupported reference bank format at {}. Expected an image file, a directory of images, "
        "or a tensor / array file shaped [N, H, W, C] or [N, C, H, W].".format(path)
    )


def _reference_array_to_tensor(reference_array, reference_obs_key):
    if torch.is_tensor(reference_array):
        reference_array = reference_array.detach().cpu().numpy()

    if reference_array.ndim == 2:
        reference_array = reference_array[..., None]

    if reference_array.ndim != 3:
        raise ValueError("Each reference image must be rank-3, got shape {}".format(reference_array.shape))

    if reference_array.shape[-1] in (1, 3):
        processed = ObsUtils.process_obs(obs=reference_array, obs_key=reference_obs_key)
        return torch.as_tensor(processed, dtype=torch.float32)

    if reference_array.shape[0] in (1, 3):
        tensor = torch.as_tensor(reference_array, dtype=torch.float32)
        if reference_array.dtype == np.uint8:
            tensor = tensor / 255.0
        return tensor

    raise ValueError(
        "Reference image shape {} is incompatible with rgb processing for obs key {}".format(
            reference_array.shape, reference_obs_key
        )
    )


def _resize_reference_tensor(reference_tensor, target_hw):
    if tuple(reference_tensor.shape[-2:]) == tuple(target_hw):
        return reference_tensor
    resized = F.interpolate(
        reference_tensor.unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def load_reference_bank(reference_bank_path, reference_obs_key, max_references, target_hw=None):
    reference_arrays = _load_reference_arrays(reference_bank_path)
    if max_references is not None:
        reference_arrays = reference_arrays[:max_references]
    if len(reference_arrays) == 0:
        raise ValueError("Reference bank at {} is empty".format(reference_bank_path))

    reference_tensors = [
        _reference_array_to_tensor(reference_array, reference_obs_key=reference_obs_key)
        for reference_array in reference_arrays
    ]
    if target_hw is not None:
        reference_tensors = [
            _resize_reference_tensor(reference_tensor, target_hw=target_hw)
            for reference_tensor in reference_tensors
        ]
    return torch.stack(reference_tensors, dim=0)


class ReferenceGatedVisualCore(ObsCore.VisualCore):
    """
    VisualCore variant that gates the current backbone feature map with cosine
    similarity to an external reference-image bank.
    """

    def __init__(
        self,
        input_shape,
        reference_bank_path,
        reference_obs_key="agentview_image",
        max_references=16,
        similarity_reduce="max",
        reference_reduce="mean",
        heatmap_activation="sigmoid",
        heatmap_scale=10.0,
        debug_store_heatmap=False,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
    ):
        super(ReferenceGatedVisualCore, self).__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=pool_class,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=pool_kwargs,
            flatten=flatten,
            feature_dimension=feature_dimension,
        )

        self.reference_obs_key = reference_obs_key
        self.similarity_reduce = similarity_reduce
        self.reference_reduce = reference_reduce
        self.heatmap_activation = heatmap_activation
        self.heatmap_scale = heatmap_scale
        self.debug_store_heatmap = debug_store_heatmap
        self.last_heatmap = None

        if reference_bank_path is None:
            raise ValueError("reference_bank_path must be set for ReferenceGatedVisualCore")

        reference_images = load_reference_bank(
            reference_bank_path=reference_bank_path,
            reference_obs_key=reference_obs_key,
            max_references=max_references,
            target_hw=input_shape[-2:],
        )
        self.register_buffer("reference_images", reference_images, persistent=False)

    def _resize_reference_images(self, inputs):
        if tuple(self.reference_images.shape[-2:]) == tuple(inputs.shape[-2:]):
            return self.reference_images
        return F.interpolate(
            self.reference_images,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _pool_reference_features(self, reference_features):
        if self.reference_reduce == "mean":
            return reference_features.mean(dim=(-1, -2))
        if self.reference_reduce == "max":
            return reference_features.amax(dim=(-1, -2))
        raise ValueError("Unsupported reference_reduce {}".format(self.reference_reduce))

    def _activate_heatmap(self, similarity_heatmap):
        if self.heatmap_activation == "sigmoid":
            return torch.sigmoid(self.heatmap_scale * similarity_heatmap)
        if self.heatmap_activation == "clamp":
            return similarity_heatmap.clamp(min=0.0, max=1.0)
        if self.heatmap_activation == "shifted_cosine":
            return ((similarity_heatmap + 1.0) / 2.0).clamp(min=0.0, max=1.0)
        if self.heatmap_activation == "relu":
            return similarity_heatmap.clamp(min=0.0)
        raise ValueError("Unsupported heatmap_activation {}".format(self.heatmap_activation))

    def forward(self, inputs):
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)

        current_features = self.backbone(inputs)
        reference_images = self._resize_reference_images(inputs).to(device=inputs.device, dtype=inputs.dtype)
        reference_features = self.backbone(reference_images)

        batch_size, channels, height, width = current_features.shape
        current_vectors = current_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        reference_vectors = self._pool_reference_features(reference_features)

        current_vectors = F.normalize(current_vectors, dim=-1)
        reference_vectors = F.normalize(reference_vectors, dim=-1)

        similarities = torch.einsum("blc,nc->bln", current_vectors, reference_vectors)
        if self.similarity_reduce == "max":
            similarity_heatmap = similarities.max(dim=-1).values
        elif self.similarity_reduce == "mean":
            similarity_heatmap = similarities.mean(dim=-1)
        else:
            raise ValueError("Unsupported similarity_reduce {}".format(self.similarity_reduce))

        similarity_heatmap = similarity_heatmap.reshape(batch_size, 1, height, width)
        heatmap = self._activate_heatmap(similarity_heatmap)
        gated_features = current_features * heatmap

        if self.debug_store_heatmap:
            self.last_heatmap = heatmap.detach()

        output = self.nets[1:](gated_features)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(output.shape)[1:]:
            raise ValueError(
                "Size mismatch: expect size {}, but got size {}".format(
                    str(self.output_shape(list(inputs.shape)[1:])),
                    str(list(output.shape)[1:]),
                )
            )
        return output


def _make_randomizers(enc_kwargs, obs_shape):
    randomizers = []
    obs_randomizer_class_list = enc_kwargs["obs_randomizer_class"]
    obs_randomizer_kwargs_list = enc_kwargs["obs_randomizer_kwargs"]

    if not isinstance(obs_randomizer_class_list, list):
        obs_randomizer_class_list = [obs_randomizer_class_list]
    if not isinstance(obs_randomizer_kwargs_list, list):
        obs_randomizer_kwargs_list = [obs_randomizer_kwargs_list]

    rand_input_shape = obs_shape
    for rand_class, rand_kwargs in zip(obs_randomizer_class_list, obs_randomizer_kwargs_list):
        rand = None
        if rand_class is not None:
            rand_kwargs = deepcopy(rand_kwargs)
            rand_kwargs["input_shape"] = rand_input_shape
            rand_kwargs = extract_class_init_kwargs_from_dict(
                cls=ObsUtils.OBS_RANDOMIZERS[rand_class],
                dic=rand_kwargs,
                copy=False,
            )
            rand = ObsUtils.OBS_RANDOMIZERS[rand_class](**rand_kwargs)
            rand_input_shape = rand.output_shape_in(rand_input_shape)
        randomizers.append(rand)
    return randomizers, rand_input_shape


def build_reference_gated_obs_encoder(
    obs_shapes,
    encoder_kwargs,
    gating_config,
    feature_activation=nn.ReLU,
):
    encoder = ObsNets.ObservationEncoder(feature_activation=feature_activation)
    gated_obs_key = gating_config.reference_obs_key

    if gating_config.enabled and gated_obs_key not in obs_shapes:
        raise ValueError("reference_obs_key {} not found in obs shapes {}".format(gated_obs_key, list(obs_shapes.keys())))

    for obs_key, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[obs_key]
        enc_kwargs = deepcopy(encoder_kwargs[obs_modality])

        if enc_kwargs.get("core_kwargs", None) is None:
            enc_kwargs["core_kwargs"] = {}
        if enc_kwargs.get("obs_randomizer_kwargs", None) is None:
            enc_kwargs["obs_randomizer_kwargs"] = {}

        randomizers, rand_input_shape = _make_randomizers(enc_kwargs=enc_kwargs, obs_shape=obs_shape)

        if gating_config.enabled and obs_key == gated_obs_key:
            core_kwargs = deepcopy(enc_kwargs["core_kwargs"])
            gated_core = ReferenceGatedVisualCore(
                input_shape=rand_input_shape,
                reference_bank_path=gating_config.reference_bank_path,
                reference_obs_key=obs_key,
                max_references=gating_config.max_references,
                similarity_reduce=gating_config.similarity_reduce,
                reference_reduce=gating_config.reference_reduce,
                heatmap_activation=gating_config.heatmap_activation,
                heatmap_scale=gating_config.heatmap_scale,
                debug_store_heatmap=gating_config.debug_store_heatmap,
                **core_kwargs
            )
            encoder.register_obs_key(
                name=obs_key,
                shape=obs_shape,
                net=gated_core,
                randomizers=randomizers,
            )
            continue

        net_class = enc_kwargs["core_class"]
        net_kwargs = deepcopy(enc_kwargs["core_kwargs"])
        if net_class is not None:
            net_kwargs["input_shape"] = rand_input_shape
            net_kwargs = extract_class_init_kwargs_from_dict(
                cls=ObsUtils.OBS_ENCODER_CORES[net_class],
                dic=net_kwargs,
                copy=False,
            )

        encoder.register_obs_key(
            name=obs_key,
            shape=obs_shape,
            net_class=net_class,
            net_kwargs=net_kwargs,
            randomizers=randomizers,
        )

    encoder.make()
    return encoder


class ReferenceGatedObservationGroupEncoder(BaseNets.Module):
    """
    ObservationGroupEncoder variant that applies reference gating to a single obs key
    without changing shared robomimic encoder behavior.
    """

    def __init__(
        self,
        observation_group_shapes,
        encoder_kwargs,
        gating_config,
        feature_activation=nn.ReLU,
    ):
        super(ReferenceGatedObservationGroupEncoder, self).__init__()

        self.observation_group_shapes = observation_group_shapes
        self.nets = nn.ModuleDict()
        for obs_group, obs_shapes in observation_group_shapes.items():
            self.nets[obs_group] = build_reference_gated_obs_encoder(
                obs_shapes=obs_shapes,
                encoder_kwargs=encoder_kwargs,
                gating_config=gating_config,
                feature_activation=feature_activation,
            )

    def forward(self, **inputs):
        assert set(self.observation_group_shapes.keys()).issubset(inputs), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        for obs_group in self.observation_group_shapes:
            outputs.append(self.nets[obs_group].forward(inputs[obs_group]))
        return torch.cat(outputs, dim=-1)

    def output_shape(self, input_shape=None):
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]


@register_algo_factory_func("diffusion_policy_ref_gating")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return DiffusionPolicyRefGatingUNet, {}
    raise RuntimeError("diffusion_policy_ref_gating currently only supports the UNet policy")


class DiffusionPolicyRefGatingUNet(DiffusionPolicyUNet):
    def _create_networks(self):
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ReferenceGatedObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
            gating_config=self.algo_config.reference_gating,
        )
        obs_encoder = replace_bn_with_gn(obs_encoder)
        obs_dim = obs_encoder.output_shape()[0]

        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim * self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net,
            })
        })
        nets = nets.float().to(self.device)

        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type,
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type,
            )
        else:
            raise RuntimeError()

        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
