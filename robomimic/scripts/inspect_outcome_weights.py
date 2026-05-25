"""
Inspect outcome labels and effective sample weights for hdf5 datasets.
"""
import argparse
import os

import h5py
import numpy as np

from robomimic.utils.dataset import compute_sequence_sample_weight


def _decode_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs_to_dict(attrs):
    return {k: _decode_attr_value(v) for k, v in attrs.items()}


def _sorted_demo_keys(demo_keys):
    try:
        return sorted(demo_keys, key=lambda elem: int(elem.split("_")[-1]))
    except ValueError:
        return sorted(demo_keys)


def _num_samples(ep_grp):
    if "num_samples" in ep_grp.attrs:
        return int(ep_grp.attrs["num_samples"])
    if "actions" in ep_grp:
        return int(ep_grp["actions"].shape[0])
    if "states" in ep_grp:
        return int(ep_grp["states"].shape[0])
    raise KeyError("Could not infer num_samples for {}".format(ep_grp.name))


def _source_from_attrs(attrs, dataset_source):
    source = attrs.get("source", dataset_source)
    return _decode_attr_value(source) if source is not None else "single"


def inspect_dataset(path, args):
    weighted_loss_config = dict(
        enabled=True,
        human_demo_weight=args.human_demo_weight,
        success_rollout_weight=args.success_rollout_weight,
        failed_rollout_weight=args.failed_rollout_weight,
        prefailure_weight=args.prefailure_weight,
        postfailure_weight=args.postfailure_weight,
        failure_window=args.failure_window,
        eps=args.eps,
    )

    counts = dict(
        single=0,
        rollout_success=0,
        rollout_failed=0,
        errors=0,
        sequences=0,
    )
    weight_sum = 0.0

    with h5py.File(os.path.expanduser(path), "r") as f:
        demos = _sorted_demo_keys(list(f["data"].keys()))
        for ep in demos:
            ep_grp = f["data/{}".format(ep)]
            attrs = _attrs_to_dict(ep_grp.attrs)
            try:
                source = _source_from_attrs(attrs, args.source)
                if source == "single":
                    counts["single"] += 1
                elif source == "rollout":
                    if "success" not in attrs:
                        raise ValueError("rollout demo is missing attrs['success']")
                    if bool(int(attrs["success"])):
                        counts["rollout_success"] += 1
                    else:
                        counts["rollout_failed"] += 1
                else:
                    raise ValueError("unexpected source '{}'".format(source))

                num_sequences = _num_samples(ep_grp)
                for index_in_demo in range(num_sequences):
                    weight_sum += compute_sequence_sample_weight(
                        demo_attrs=attrs,
                        index_in_demo=index_in_demo,
                        seq_length=args.seq_length,
                        weighted_loss_config=weighted_loss_config,
                        dataset_source=args.source,
                        dataset_path=path,
                        demo_id=ep,
                    )
                counts["sequences"] += num_sequences

            except Exception as e:
                counts["errors"] += 1
                print("{} {}: ERROR: {}".format(path, ep, e))

    avg_weight = weight_sum / max(counts["sequences"], 1)
    print("")
    print("Dataset: {}".format(path))
    print("  single demos: {}".format(counts["single"]))
    print("  rollout success demos: {}".format(counts["rollout_success"]))
    print("  rollout failed demos: {}".format(counts["rollout_failed"]))
    print("  errors: {}".format(counts["errors"]))
    print("  sampled sequences: {}".format(counts["sequences"]))
    print("  effective average sample weight: {:.6f}".format(avg_weight))

    return counts, weight_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="hdf5 dataset path(s) to inspect",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["single", "rollout"],
        default=None,
        help="optional dataset-level source override",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=16,
        help="sequence length used to estimate per-sequence weights",
    )
    parser.add_argument("--human_demo_weight", type=float, default=1.0)
    parser.add_argument("--success_rollout_weight", type=float, default=1.0)
    parser.add_argument("--failed_rollout_weight", type=float, default=0.1)
    parser.add_argument("--prefailure_weight", type=float, default=0.3)
    parser.add_argument("--postfailure_weight", type=float, default=0.0)
    parser.add_argument("--failure_window", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-8)
    parsed_args = parser.parse_args()

    total_counts = dict(single=0, rollout_success=0, rollout_failed=0, errors=0, sequences=0)
    total_weight_sum = 0.0
    for dataset_path in parsed_args.datasets:
        dataset_counts, dataset_weight_sum = inspect_dataset(dataset_path, parsed_args)
        total_weight_sum += dataset_weight_sum
        for key in total_counts:
            total_counts[key] += dataset_counts[key]

    if len(parsed_args.datasets) > 1:
        avg_weight = total_weight_sum / max(total_counts["sequences"], 1)
        print("")
        print("Total")
        print("  single demos: {}".format(total_counts["single"]))
        print("  rollout success demos: {}".format(total_counts["rollout_success"]))
        print("  rollout failed demos: {}".format(total_counts["rollout_failed"]))
        print("  errors: {}".format(total_counts["errors"]))
        print("  sampled sequences: {}".format(total_counts["sequences"]))
        print("  effective average sample weight: {:.6f}".format(avg_weight))
