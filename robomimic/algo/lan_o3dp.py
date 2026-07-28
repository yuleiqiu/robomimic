"""LAN-O3DP policy registration.

This class deliberately reuses the standard Diffusion Policy DDPM training,
EMA, inference queues, and action interface. Only LAN-specific observation
encoding behavior is changed here.
"""

from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


@register_algo_factory_func("lan_o3dp")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return LanO3DPUNet, {}
    if algo_config.transformer.enabled:
        raise NotImplementedError()
    raise RuntimeError()


class LanO3DPUNet(DiffusionPolicyUNet):
    """Unguided LAN-O3DP-compatible point-cloud diffusion policy."""

    def _obs_encoder_feature_activation(self):
        # Preserve the point encoder's final Linear + LayerNorm output. The
        # public LAN implementation does not add robomimic's extra ReLU.
        return None
