import pytest

import robomimic.utils.file_utils as FileUtils


def test_select_checkpoint_metadata_dict():
    metadata = {"all_shapes": {}, "ac_dim": 7}

    selected = FileUtils._select_checkpoint_metadata(metadata, "shape_metadata")

    assert selected is metadata


def test_select_checkpoint_metadata_list():
    metadata = [
        {"all_shapes": {"obs": [3]}, "ac_dim": 7},
        {"all_shapes": {"obs": [4]}, "ac_dim": 8},
    ]

    selected = FileUtils._select_checkpoint_metadata(metadata, "shape_metadata")

    assert selected is metadata[0]


def test_select_checkpoint_metadata_empty_list():
    with pytest.raises(ValueError, match="shape_metadata.*empty list"):
        FileUtils._select_checkpoint_metadata([], "shape_metadata")
