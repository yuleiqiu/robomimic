"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""
import os
import json
import sys
import numpy as np
from datetime import datetime
from contextlib import contextmanager
import textwrap
import time
from tqdm import tqdm
from termcolor import colored

import robomimic

# global list of warning messages can be populated with @log_warning and flushed with @flush_warnings
WARNINGS_BUFFER = []


class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # ensure stdout gets flushed
        self.terminal.flush()


class DataLogger(object):
    """
    Logging class to log metrics to tensorboard and/or retrieve running statistics about logged data.
    """
    def __init__(self, log_dir, config, log_tb=True, log_wandb=False):
        """
        Args:
            log_dir (str): base path to store logs
            log_tb (bool): whether to use tensorboard logging
        """
        self._tb_logger = None
        self._wandb_logger = None
        self._wandb_run = None
        self._wandb_required = bool(
            config.experiment.logging.get("wandb_required", False)
        )
        self._wandb_manifest_path = os.path.join(log_dir, "wandb_run.json")
        self._wandb_manifest = None
        self._data = dict() # store all the scalar data logged so far

        if self._wandb_required and not log_wandb:
            raise ValueError("wandb_required=true requires log_wandb=true")

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

        if log_wandb:
            import wandb
            import robomimic.macros as Macros
            
            # set up wandb api key if specified in macros
            if Macros.WANDB_API_KEY is not None:
                os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

            assert Macros.WANDB_ENTITY is not None, "WANDB_ENTITY macro is set to None." \
                    "\nSet this macro in {base_path}/macros_private.py" \
                    "\nIf this file does not exist, first run python {base_path}/scripts/setup_macros.py".format(base_path=robomimic.__path__[0])
            
            # Required experiment runs fail fast and never silently switch to
            # offline mode. Historical optional runs retain retry + fallback.
            num_attempts = 1 if self._wandb_required else 10
            for attempt in range(num_attempts):
                try:
                    # set up wandb
                    self._wandb_logger = wandb

                    self._wandb_run = self._wandb_logger.init(
                        entity=Macros.WANDB_ENTITY,
                        project=config.experiment.logging.wandb_proj_name,
                        name=config.experiment.name,
                        dir=log_dir,
                        mode=(
                            "online"
                            if self._wandb_required or attempt < num_attempts - 1
                            else "offline"
                        ),
                    )
                    if self._wandb_run is None:
                        raise RuntimeError("wandb.init returned no active run")

                    # set up info for identifying experiment
                    wandb_config = {k: v for (k, v) in config.meta.items() if k not in ["hp_keys", "hp_values"]}
                    for (k, v) in zip(config.meta["hp_keys"], config.meta["hp_values"]):
                        wandb_config[k] = v
                    if "algo" not in wandb_config:
                        wandb_config["algo"] = config.algo_name
                    wandb_config["robomimic_config"] = json.loads(
                        json.dumps(config)
                    )
                    self._wandb_logger.config.update(wandb_config)

                    self._wandb_manifest = {
                        "run_id": self._wandb_run.id,
                        "run_url": self._wandb_run.url,
                        "entity": Macros.WANDB_ENTITY,
                        "project": config.experiment.logging.wandb_proj_name,
                        "name": config.experiment.name,
                        "mode": "online" if self._wandb_required else self._wandb_run.settings.mode,
                        "required": self._wandb_required,
                        "config": json.loads(json.dumps(config)),
                        "checkpoints": [],
                    }
                    self._write_wandb_manifest()

                    break
                except Exception as e:
                    log_warning("wandb initialization error (attempt #{}): {}".format(attempt + 1, e))
                    self._wandb_logger = None
                    self._wandb_run = None
                    if self._wandb_required:
                        raise RuntimeError(
                            "Required online W&B initialization failed"
                        ) from e
                    time.sleep(30)

    def _write_wandb_manifest(self):
        if self._wandb_manifest is None:
            return
        temporary = self._wandb_manifest_path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(self._wandb_manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, self._wandb_manifest_path)

    def record_checkpoint(self, path, epoch, kind):
        """Persist the W&B run-to-checkpoint correspondence."""

        if self._wandb_manifest is None:
            return
        self._wandb_manifest["checkpoints"].append(
            {
                "path": os.path.abspath(path),
                "epoch": int(epoch),
                "kind": str(kind),
            }
        )
        self._write_wandb_manifest()
        try:
            self._wandb_logger.config.update(
                {"checkpoint_map": self._wandb_manifest["checkpoints"]},
                allow_val_change=True,
            )
        except Exception as error:
            if self._wandb_required:
                raise RuntimeError(
                    "Required online W&B checkpoint mapping failed"
                ) from error
            log_warning("wandb checkpoint mapping: {}".format(error))

    def record(self, k, v, epoch, data_type='scalar', log_stats=False):
        """
        Record data with logger.
        Args:
            k (str): key string
            v (float or image): value to store
            epoch: current epoch number
            data_type (str): the type of data. either 'scalar' or 'image'
            log_stats (bool): whether to store the mean/max/min/std for all data logged so far with key k
        """

        assert data_type in ['scalar', 'image']

        if data_type == 'scalar':
            # maybe update internal cache if logging stats for this key
            if log_stats or k in self._data: # any key that we're logging or previously logged
                if k not in self._data:
                    self._data[k] = []
                self._data[k].append(v)

        # maybe log to tensorboard
        if self._tb_logger is not None:
            if data_type == 'scalar':
                self._tb_logger.add_scalar(k, v, epoch)
                if log_stats:
                    stats = self.get_stats(k)
                    for (stat_k, stat_v) in stats.items():
                        stat_k_name = '{}-{}'.format(k, stat_k)
                        self._tb_logger.add_scalar(stat_k_name, stat_v, epoch)
            elif data_type == 'image':
                if len(v.shape) == 3:
                    v = v[None, ...]
                self._tb_logger.add_images(k, img_tensor=v, global_step=epoch, dataformats="NHWC")

        if self._wandb_logger is not None:
            try:
                if data_type == 'scalar':
                    self._wandb_logger.log({k: v}, step=epoch)
                    if log_stats:
                        stats = self.get_stats(k)
                        for (stat_k, stat_v) in stats.items():
                            self._wandb_logger.log({"{}/{}".format(k, stat_k): stat_v}, step=epoch)
                elif data_type == 'image':
                    import wandb
                    self._wandb_logger.log({k: wandb.Image(v)}, step=epoch)
            except Exception as e:
                if self._wandb_required:
                    raise RuntimeError("Required online W&B logging failed") from e
                log_warning("wandb logging: {}".format(e))

    def get_stats(self, k):
        """
        Computes running statistics for a particular key.
        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats['mean'] = np.mean(self._data[k])
        stats['std'] = np.std(self._data[k])
        stats['min'] = np.min(self._data[k])
        stats['max'] = np.max(self._data[k])
        return stats

    def close(self):
        """
        Run before terminating to make sure all logs are flushed
        """
        if self._tb_logger is not None:
            self._tb_logger.close()

        if self._wandb_logger is not None:
            self._wandb_logger.finish()


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


@contextmanager
def silence_stdout():
    """
    This contextmanager will redirect stdout so that nothing is printed
    to the terminal. Taken from the link below:

    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    """
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


def log_warning(message, color="yellow", print_now=True):
    """
    This function logs a warning message by recording it in a global warning buffer.
    The global registry will be maintained until @flush_warnings is called, at
    which point the warnings will get printed to the terminal.

    Args:
        message (str): warning message to display
        color (str): color of message - defaults to "yellow"
        print_now (bool): if True (default), will print to terminal immediately, in
            addition to adding it to the global warning buffer
    """
    global WARNINGS_BUFFER
    buffer_message = colored("ROBOMIMIC WARNING(\n{}\n)".format(textwrap.indent(message, "    ")), color)
    WARNINGS_BUFFER.append(buffer_message)
    if print_now:
        print(buffer_message)


def flush_warnings():
    """
    This function flushes all warnings from the global warning buffer to the terminal and
    clears the global registry.
    """
    global WARNINGS_BUFFER
    for msg in WARNINGS_BUFFER:
        print(msg)
    WARNINGS_BUFFER = []
