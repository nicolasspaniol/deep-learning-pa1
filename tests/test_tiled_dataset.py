from test_bbbc038_dataset import dataset_path

import pytest
import torch

from bbbc038_dataset import BBBC038Dataset
from tiled_dataset import TiledBBBC038Dataset


# --- Length -----------------------------------------------------------

@pytest.mark.parametrize("n_tiles", [1, 2, 3])
def test_len_scales_by_n_tiles_squared(dataset_path, n_tiles):
    base = BBBC038Dataset(is_training=False, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=n_tiles, overlap=0.0)

    assert len(tiled) == len(base) * n_tiles ** 2


# --- Tile bounds (pure logic, no I/O) ----------------------------------

def test_tile_bounds_partition_image_with_no_overlap(dataset_path):
    base = BBBC038Dataset(is_training=False, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)

    bounds = tiled._tile_bounds(4, 4)

    assert bounds == [(0, 2, 0, 2), (0, 2, 2, 4), (2, 4, 0, 2), (2, 4, 2, 4)]


def test_tile_bounds_grow_with_overlap(dataset_path):
    base = BBBC038Dataset(is_training=False, path=dataset_path, img_size=4)
    tiled_no_overlap = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)
    tiled_overlap = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.5)

    plain = tiled_no_overlap._tile_bounds(8, 8)
    overlapped = tiled_overlap._tile_bounds(8, 8)

    # every overlapped tile should be at least as large as its plain counterpart
    for (y0, y1, x0, x1), (oy0, oy1, ox0, ox1) in zip(plain, overlapped):
        assert oy0 <= y0 and oy1 >= y1
        assert ox0 <= x0 and ox1 >= x1


# --- __getitem__, training (only touches sample-a, which has masks) ----

def test_training_getitem_shapes(dataset_path):
    base = BBBC038Dataset(is_training=True, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)

    # indices 0-3 all map back to sample_idx 0 ("sample-a"), which has masks
    tile_id, image, semantic_mask, instance_map = tiled[0]

    assert tile_id == "sample-a_tile0"
    assert image.shape == (3, 4, 4)
    assert semantic_mask.shape == (4, 4)
    assert instance_map.shape == (4, 4)
    assert image.dtype == torch.float32
    assert instance_map.dtype == torch.int64


def test_training_tile_ids_are_unique(dataset_path):
    base = BBBC038Dataset(is_training=True, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)

    tile_ids = [tiled[i][0] for i in range(4)]

    assert tile_ids == [
        "sample-a_tile0",
        "sample-a_tile1",
        "sample-a_tile2",
        "sample-a_tile3",
    ]


def test_tiles_preserve_correct_instance_labels(dataset_path):
    # img_size == raw size and overlap=0 makes the quadrants predictable:
    # first_mask (id 1) sits top-left, second_mask (id 2) sits bottom-right.
    base = BBBC038Dataset(is_training=True, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)

    _, _, _, top_left_instance = tiled[0]
    _, _, _, bottom_right_instance = tiled[3]

    assert torch.all(top_left_instance == 1)
    assert torch.all(bottom_right_instance == 2)


# --- __getitem__, inference (safe for sample-b, which has no masks) ----

def test_inference_getitem_returns_two_tuple(dataset_path):
    base = BBBC038Dataset(is_training=False, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)

    item = tiled[0]

    assert len(item) == 2
    tile_id, image = item
    assert tile_id == "sample-a_tile0"
    assert image.shape == (3, 4, 4)


def test_inference_covers_sample_without_masks(dataset_path):
    base = BBBC038Dataset(is_training=False, path=dataset_path, img_size=4)
    tiled = TiledBBBC038Dataset(base, n_tiles=2, overlap=0.0)

    # sample-b is the second id, so its tiles start at index n_tiles**2
    tile_id, image = tiled[4]

    assert tile_id == "sample-b_tile0"
    assert image.shape == (3, 4, 4)
