"""
Split an annotated hdf5 dataset into success and failure datasets.

This script expects each demonstration to have a per-demo success attribute,
such as the one written by annotate_dataset_success.py.

Example usage:

    python split_dataset_by_success.py --dataset /path/to/rollouts.hdf5

Then create train / validation masks in each output dataset:

    python split_train_val.py --dataset /path/to/rollouts_success.hdf5 --ratio 0.1
    python split_train_val.py --dataset /path/to/rollouts_failure.hdf5 --ratio 0.1
"""
import argparse
import os

import h5py
import numpy as np


def _sorted_demo_keys(demo_keys):
    """
    Sort demo keys numerically when they follow the usual demo_<int> format.
    """
    try:
        return sorted(demo_keys, key=lambda elem: int(elem.split("_")[-1]))
    except ValueError:
        return sorted(demo_keys)


def _copy_attrs(src, dst):
    """
    Copy hdf5 attributes from one object to another.
    """
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _default_output_paths(dataset_path):
    """
    Construct default success and failure output paths from the source path.
    """
    dataset_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    base, ext = os.path.splitext(dataset_name)
    if ext == "":
        ext = ".hdf5"
    return (
        os.path.join(dataset_dir, "{}_success{}".format(base, ext)),
        os.path.join(dataset_dir, "{}_failure{}".format(base, ext)),
    )


def _check_output_path(path, overwrite):
    """
    Fail early if an output path exists and overwrite is not enabled.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            "Output file already exists: {}. Pass --overwrite to replace it.".format(path)
        )


def _demo_success(ep_grp, success_attr):
    """
    Read a success attribute from a demo group as a bool.
    """
    if success_attr not in ep_grp.attrs:
        raise KeyError(
            "Demo {} is missing attrs[{}]. Run annotate_dataset_success.py first.".format(
                ep_grp.name,
                success_attr,
            )
        )
    return bool(ep_grp.attrs[success_attr])


def _num_samples(ep_grp):
    """
    Get the number of samples for a demo.
    """
    if "num_samples" in ep_grp.attrs:
        return int(ep_grp.attrs["num_samples"])
    if "actions" in ep_grp:
        return int(ep_grp["actions"].shape[0])
    if "states" in ep_grp:
        return int(ep_grp["states"].shape[0])
    raise KeyError("Could not infer num_samples for {}".format(ep_grp.name))


def _write_subset(src_file, output_path, demo_keys, overwrite):
    """
    Write a subset of demos to a new hdf5 file.
    """
    mode = "w" if overwrite else "x"
    total_samples = 0

    with h5py.File(output_path, mode) as f_out:
        _copy_attrs(src_file, f_out)

        data_grp = f_out.create_group("data")
        _copy_attrs(src_file["data"], data_grp)

        for ep in demo_keys:
            src_ep_grp = src_file["data/{}".format(ep)]
            src_file.copy(src_ep_grp, data_grp, name=ep)
            total_samples += _num_samples(src_ep_grp)

        data_grp.attrs["total"] = total_samples


def split_dataset_by_success(args):
    if args.success_path is None or args.failure_path is None:
        default_success_path, default_failure_path = _default_output_paths(args.dataset)
        success_path = args.success_path or default_success_path
        failure_path = args.failure_path or default_failure_path
    else:
        success_path = args.success_path
        failure_path = args.failure_path

    _check_output_path(success_path, overwrite=args.overwrite)
    _check_output_path(failure_path, overwrite=args.overwrite)

    with h5py.File(args.dataset, "r") as f:
        if args.filter_key is not None:
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
        else:
            demos = list(f["data"].keys())
        demos = _sorted_demo_keys(demos)

        success_demos = []
        failure_demos = []
        for ep in demos:
            ep_grp = f["data/{}".format(ep)]
            if _demo_success(ep_grp=ep_grp, success_attr=args.success_attr):
                success_demos.append(ep)
            else:
                failure_demos.append(ep)

        _write_subset(
            src_file=f,
            output_path=success_path,
            demo_keys=success_demos,
            overwrite=args.overwrite,
        )
        _write_subset(
            src_file=f,
            output_path=failure_path,
            demo_keys=failure_demos,
            overwrite=args.overwrite,
        )

    print("Source dataset: {}".format(args.dataset))
    print("Success demos: {} -> {}".format(len(success_demos), success_path))
    print("Failure demos: {} -> {}".format(len(failure_demos), failure_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to annotated hdf5 dataset",
    )
    parser.add_argument(
        "--success_path",
        type=str,
        default=None,
        help="output path for successful demos; defaults to <dataset>_success.hdf5",
    )
    parser.add_argument(
        "--failure_path",
        type=str,
        default=None,
        help="output path for failed demos; defaults to <dataset>_failure.hdf5",
    )
    parser.add_argument(
        "--success_attr",
        type=str,
        default="success",
        help="per-demo hdf5 attribute name containing success labels",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="optional mask key specifying which demos to split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite output files if they already exist",
    )

    split_dataset_by_success(parser.parse_args())
