from test_bbbc038_dataset import dataset_path

import torch

from src.bbbc038_dataset import BBBC038Dataset
from src.tiled_dataset import TiledBBBC038Dataset


IMG_SIZE = 128
TILE_SIZE = 48


def make_base(dataset_path, is_training):
    return BBBC038Dataset(
        is_training=is_training, path=dataset_path, img_size=IMG_SIZE
    )


# --- Fixed tiling configuration ---------------------------------------

def test_len_scales_by_sixteen(dataset_path):
    base = make_base(dataset_path, is_training=False)
    tiled = TiledBBBC038Dataset(base)

    assert tiled.n_tiles == 4
    assert tiled.stride == 32
    assert tiled.tile_size == TILE_SIZE
    assert len(tiled) == len(base) * 16


# --- Tile bounds (pure logic, no I/O) ---------------------------------

def test_tile_bounds_are_uniform_overlapping_windows(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=False))

    assert [tiled._tile_bounds(tile_idx) for tile_idx in range(16)] == [
        (0, 48, 0, 48), (0, 48, 24, 72), (0, 48, 56, 104), (0, 48, 80, 128),
        (24, 72, 0, 48), (24, 72, 24, 72), (24, 72, 56, 104), (24, 72, 80, 128),
        (56, 104, 0, 48), (56, 104, 24, 72), (56, 104, 56, 104), (56, 104, 80, 128),
        (80, 128, 0, 48), (80, 128, 24, 72), (80, 128, 56, 104), (80, 128, 80, 128),
    ]


def test_tile_bounds_keep_edge_tiles_at_full_size(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=False))

    for y0, y1, x0, x1 in (tiled._tile_bounds(i) for i in range(16)):
        assert y1 - y0 == TILE_SIZE
        assert x1 - x0 == TILE_SIZE


# --- __getitem__, training (only touches sample-a, which has masks) ---

def test_training_getitem_shapes(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=True))

    tile_id, image, semantic_mask, instance_map = tiled[0]

    assert tile_id == "sample-a_tile0"
    assert image.shape == (3, TILE_SIZE, TILE_SIZE)
    assert semantic_mask.shape == (TILE_SIZE, TILE_SIZE)
    assert instance_map.shape == (TILE_SIZE, TILE_SIZE)
    assert image.dtype == torch.float32
    assert instance_map.dtype == torch.int64


def test_training_tile_ids_are_unique(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=True))

    tile_ids = [tiled[i][0] for i in range(16)]

    assert tile_ids == [f"sample-a_tile{i}" for i in range(16)]


def test_tiles_preserve_correct_instance_labels(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=True))

    _, _, _, top_left_instance = tiled[0]
    _, _, _, bottom_right_instance = tiled[15]

    assert torch.all(top_left_instance == 1)
    assert torch.all(bottom_right_instance == 2)


# --- __getitem__, inference (safe for sample-b, which has no masks) ---

def test_inference_getitem_returns_two_tuple(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=False))

    item = tiled[0]

    assert len(item) == 2
    tile_id, image = item
    assert tile_id == "sample-a_tile0"
    assert image.shape == (3, TILE_SIZE, TILE_SIZE)


def test_inference_covers_sample_without_masks(dataset_path):
    tiled = TiledBBBC038Dataset(make_base(dataset_path, is_training=False))

    # sample-b is the second id, so its tiles start at index 16.
    tile_id, image = tiled[16]

    assert tile_id == "sample-b_tile0"
    assert image.shape == (3, TILE_SIZE, TILE_SIZE)
