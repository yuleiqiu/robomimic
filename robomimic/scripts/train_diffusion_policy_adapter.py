"""
Training entrypoint for adapter-based Diffusion Policy with dual data loaders.
"""

import argparse
from copy import deepcopy
import json
import numpy as np
import os
import psutil
import shutil
import sys
import time
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger, custom_tqdm, flush_warnings


def _normalize_data_config(data):
    if data is None:
        return None
    if isinstance(data, str):
        return [{"path": data}]
    return list(data)


def _make_domain_config(config, data_cfgs, train_filter_key, valid_filter_key):
    domain_config = deepcopy(config)
    with domain_config.values_unlocked():
        domain_config.train.data = deepcopy(data_cfgs)
        domain_config.train.hdf5_filter_key = train_filter_key
        domain_config.train.hdf5_validation_filter_key = valid_filter_key
    return domain_config


def _collect_env_and_shape_metadata(config, data_cfgs):
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in data_cfgs:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        from robomimic.utils.python_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True,
        )
        shape_meta_list.append(shape_meta)

    return env_meta_list, shape_meta_list


def _assert_shape_meta_compatible(reference_shape_meta, other_shape_meta, domain_name):
    if reference_shape_meta["ac_dim"] != other_shape_meta["ac_dim"]:
        raise ValueError(
            "action dimension mismatch between single dataset and {} dataset: {} vs {}".format(
                domain_name,
                reference_shape_meta["ac_dim"],
                other_shape_meta["ac_dim"],
            )
        )
    if reference_shape_meta["all_shapes"] != other_shape_meta["all_shapes"]:
        raise ValueError("observation shape mismatch between single dataset and {} dataset".format(domain_name))


def _load_domain_data_for_training(config, data_cfgs, obs_keys, train_filter_key, valid_filter_key):
    domain_config = _make_domain_config(
        config=config,
        data_cfgs=data_cfgs,
        train_filter_key=train_filter_key,
        valid_filter_key=valid_filter_key,
    )
    return TrainUtils.load_data_for_training(domain_config, obs_keys=obs_keys)


def _make_loader(dataset, sampler, batch_size, num_workers):
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        drop_last=True,
    )


def _next_from_loader(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def run_epoch_dual(
    model,
    single_loader,
    multi_loader,
    epoch,
    validate=False,
    num_steps=None,
    obs_normalization_stats=None,
):
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = min(len(single_loader), len(multi_loader))

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])

    single_loader_iter = iter(single_loader)
    multi_loader_iter = iter(multi_loader)

    for _ in custom_tqdm(range(num_steps)):
        t = time.time()
        single_batch, single_loader_iter = _next_from_loader(single_loader_iter, single_loader)
        multi_batch, multi_loader_iter = _next_from_loader(multi_loader_iter, multi_loader)
        timing_stats["Data_Loading"].append(time.time() - t)

        t = time.time()
        input_batch = {
            "single": model.process_batch_for_training(single_batch),
            "multi": model.process_batch_for_training(multi_batch),
        }
        input_batch = model.postprocess_batch_for_training(
            input_batch,
            obs_normalization_stats=obs_normalization_stats,
        )
        timing_stats["Process_Batch"].append(time.time() - t)

        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)
        model.on_gradient_step()

        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    step_log_dict = {}
    for step_log in step_log_all:
        for key, value in step_log.items():
            if key not in step_log_dict:
                step_log_dict[key] = []
            step_log_dict[key].append(value)
    step_log_all = dict((key, float(np.mean(values))) for key, values in step_log_dict.items())

    for key in timing_stats:
        step_log_all["Time_{}".format(key)] = np.sum(timing_stats[key]) / 60.0
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.0

    return step_log_all


def train(config, device, resume=False):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=resume)

    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    single_data_cfgs = _normalize_data_config(config.train.data)
    multi_data_cfgs = _normalize_data_config(config.train.multi_data)
    if not single_data_cfgs:
        raise ValueError("config.train.data must specify the single-domain dataset")
    if not multi_data_cfgs:
        raise ValueError("config.train.multi_data must specify the few-shot multi-domain dataset")

    single_env_meta_list, single_shape_meta_list = _collect_env_and_shape_metadata(config, single_data_cfgs)
    multi_env_meta_list, multi_shape_meta_list = _collect_env_and_shape_metadata(config, multi_data_cfgs)
    for shape_meta in multi_shape_meta_list:
        _assert_shape_meta_compatible(single_shape_meta_list[0], shape_meta, domain_name="multi")

    env_meta_list = single_env_meta_list
    shape_meta_list = single_shape_meta_list

    if config.experiment.env is not None:
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        for env_i in range(len(env_meta_list)):
            dataset_cfg = single_data_cfgs[env_i]
            do_eval = dataset_cfg.get("eval", True)
            if not do_eval:
                continue

            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]
            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            def create_env(env_name):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                env = EnvUtils.wrap_env_from_config(env, config=config)
                return env

            for env_name in env_names:
                env = create_env(env_name)
                env_key = os.path.splitext(os.path.basename(dataset_cfg["path"]))[0] if not dataset_cfg.get("key", None) else dataset_cfg["key"]
                envs[env_key] = env
                print(env)

    print("")

    single_trainset, single_validset = _load_domain_data_for_training(
        config=config,
        data_cfgs=single_data_cfgs,
        obs_keys=shape_meta_list[0]["all_obs_keys"],
        train_filter_key=config.train.hdf5_filter_key,
        valid_filter_key=config.train.hdf5_validation_filter_key,
    )
    multi_trainset, multi_validset = _load_domain_data_for_training(
        config=config,
        data_cfgs=multi_data_cfgs,
        obs_keys=shape_meta_list[0]["all_obs_keys"],
        train_filter_key=config.train.multi_hdf5_filter_key,
        valid_filter_key=config.train.multi_hdf5_validation_filter_key,
    )

    print("\n============= Single Training Dataset =============")
    print(single_trainset)
    print("")
    print("\n============= Multi Training Dataset =============")
    print(multi_trainset)
    print("")
    if single_validset is not None:
        print("\n============= Single Validation Dataset =============")
        print(single_validset)
        print("")
    if multi_validset is not None:
        print("\n============= Multi Validation Dataset =============")
        print(multi_validset)
        print("")

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = single_trainset.get_obs_normalization_stats()

    action_normalization_stats = single_trainset.get_action_normalization_stats()
    multi_trainset.set_action_normalization_stats(action_normalization_stats)
    if single_validset is not None:
        single_validset.set_action_normalization_stats(action_normalization_stats)
    if multi_validset is not None:
        multi_validset.set_action_normalization_stats(action_normalization_stats)

    single_train_sampler = single_trainset.get_dataset_sampler()
    multi_train_sampler = multi_trainset.get_dataset_sampler()
    single_batch_size = config.train.batch_size
    multi_batch_size = config.train.multi_batch_size or config.train.batch_size
    single_num_workers = config.train.num_data_workers
    multi_num_workers = config.train.multi_num_data_workers
    if multi_num_workers is None:
        multi_num_workers = config.train.num_data_workers

    single_train_loader = _make_loader(
        dataset=single_trainset,
        sampler=single_train_sampler,
        batch_size=single_batch_size,
        num_workers=single_num_workers,
    )
    multi_train_loader = _make_loader(
        dataset=multi_trainset,
        sampler=multi_train_sampler,
        batch_size=multi_batch_size,
        num_workers=multi_num_workers,
    )

    if config.experiment.validate:
        single_valid_sampler = single_validset.get_dataset_sampler()
        multi_valid_sampler = multi_validset.get_dataset_sampler()
        valid_single_workers = min(single_num_workers, 1)
        valid_multi_workers = min(multi_num_workers, 1)
        single_valid_loader = _make_loader(
            dataset=single_validset,
            sampler=single_valid_sampler,
            batch_size=single_batch_size,
            num_workers=valid_single_workers,
        )
        multi_valid_loader = _make_loader(
            dataset=multi_validset,
            sampler=multi_valid_sampler,
            batch_size=multi_batch_size,
            num_workers=valid_multi_workers,
        )
    else:
        single_valid_loader = None
        multi_valid_loader = None

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    with config.values_unlocked():
        if "optim_params" in config.algo:
            train_batches = min(len(single_trainset), len(multi_trainset))
            if train_num_steps is not None:
                train_batches = train_num_steps
            for key in config.algo.optim_params:
                config.algo.optim_params[key]["num_train_batches"] = train_batches
                config.algo.optim_params[key]["num_epochs"] = config.train.num_epochs

    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    if resume:
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as exc:
            print("got error: {} when loading from {}".format(exc, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)
    else:
        ckpt_path = config.experiment.ckpt_path
        if ckpt_path is not None:
            print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
            ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
            model.deserialize(ckpt_dict["model"])
        elif config.algo.adapter.base_ckpt_path is not None:
            print("INITIALIZING FROZEN BASE POLICY FROM " + config.algo.adapter.base_ckpt_path)
            base_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=config.algo.adapter.base_ckpt_path)
            model.load_base_policy_from_checkpoint(base_ckpt_dict)

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")

    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    best_valid_loss = None
    best_return = {key: -np.inf for key in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {key: -1.0 for key in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    start_epoch = 1
    if resume:
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = run_epoch_dual(
            model=model,
            single_loader=single_train_loader,
            multi_loader=multi_train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (
                config.experiment.save.every_n_seconds is not None
                and time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                config.experiment.save.every_n_epochs is not None
                and epoch > 0
                and epoch % config.experiment.save.every_n_epochs == 0
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for key, value in step_log.items():
            if key.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(key[5:]), value, epoch)
            else:
                data_logger.record("Train/{}".format(key), value, epoch)

        if config.experiment.validate:
            with torch.no_grad():
                step_log = run_epoch_dual(
                    model=model,
                    single_loader=single_valid_loader,
                    multi_loader=multi_valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for key, value in step_log.items():
                if key.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(key[5:]), value, epoch)
                else:
                    data_logger.record("Valid/{}".format(key), value, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or step_log["Loss"] <= best_valid_loss):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=config.experiment.rollout.n,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for key, value in rollout_logs.items():
                    if key.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, key[5:]), value, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(key, env_name), value, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print("Env: {}".format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
        )

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
            shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )

        shutil.copyfile(latest_model_path, latest_model_backup_path)
        print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    data_logger.close()


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        algo_name = args.algo if args.algo is not None else "diffusion_policy_adapter"
        config = config_factory(algo_name)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]
    if args.multi_dataset is not None:
        config.train.multi_data = [{"path": args.multi_dataset}]
    if args.base_ckpt is not None:
        config.algo.adapter.base_ckpt_path = args.base_ckpt

    if args.name is not None:
        config.experiment.name = args.name

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    if args.debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.train.output_dir = "/tmp/tmp_trained_models"

    config.lock()

    res_str = "finished run successfully!"
    try:
        train(config, device=device, resume=args.resume)
    except Exception as exc:
        res_str = "run failed with error:\n{}\n\n{}".format(exc, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--multi-dataset", type=str, default=None)
    parser.add_argument("--base-ckpt", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
