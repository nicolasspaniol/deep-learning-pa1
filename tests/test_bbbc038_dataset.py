import cv2
import numpy as np
import pytest
import torch

from bbbc038_dataset import BBBC038Dataset
from utils import is_binary

@pytest.fixture
def dataset_path(tmp_path):
    """Create two minimal BBBC038-style samples without relying on downloads."""
    sample_a = tmp_path / "sample-a"
    images_a = sample_a / "images"
    masks_a = sample_a / "masks"
    images_a.mkdir(parents=True)
    masks_a.mkdir()

    # OpenCV writes BGRA. The alpha channel makes sure the dataset discards it.
    image_a = np.full((4, 4, 4), [30, 20, 10, 99], dtype=np.uint8)
    first_mask = np.zeros((4, 4), dtype=np.uint8)
    second_mask = np.zeros((4, 4), dtype=np.uint8)
    first_mask[:2, :2] = 255
    second_mask[2:, 2:] = 255
    assert cv2.imwrite(str(images_a / "sample-a.png"), image_a)
    assert cv2.imwrite(str(masks_a / "a-first.png"), first_mask)
    assert cv2.imwrite(str(masks_a / "z-second.png"), second_mask)

    # A test-set sample intentionally has no masks directory.
    sample_b = tmp_path / "sample-b"
    images_b = sample_b / "images"
    images_b.mkdir(parents=True)
    assert cv2.imwrite(
        str(images_b / "sample-b.png"),
        np.full((4, 4, 3), [60, 50, 40], dtype=np.uint8),
    )

    # Dataset roots can contain non-sample artifacts.
    (tmp_path / "README.txt").write_text("not a sample")
    return tmp_path


def test_bbbc038_dataset_requires_existing_data(tmp_path):
    with pytest.raises(RuntimeError, match="Dataset not found"):
        BBBC038Dataset(True, tmp_path / 'fake_directory')


def test_bbbc038_dataset_with_empty_directory_is_empty(tmp_path):
    assert len(BBBC038Dataset(True, tmp_path)) == 0


def test_bbbc038_dataset_discovers_only_sorted_sample_directories(dataset_path):
    dataset = BBBC038Dataset(is_training=True, path=dataset_path)

    assert len(dataset) == 2
    assert dataset.ids == ["sample-a", "sample-b"]


def test_bbbc038_training_sample_returns_resized_rgb_image(dataset_path):
    dataset = BBBC038Dataset(is_training=True, path=dataset_path, img_size=8)

    sample_id, image, _, _ = dataset[0]

    assert sample_id == "sample-a"
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert torch.all((image >= 0) & (image <= 1))
    assert torch.allclose(image[:, 0, 0], torch.tensor([10, 20, 30]) / 255)


def test_bbbc038_training_sample_returns_binary_semantic_mask(dataset_path):
    dataset = BBBC038Dataset(is_training=True, path=dataset_path, img_size=8)

    _, _, semantic_mask, instance_map = dataset[0]

    assert semantic_mask.shape == (8, 8)
    assert semantic_mask.dtype == torch.float32
    assert is_binary(semantic_mask)
    assert torch.equal(semantic_mask, (instance_map > 0).float())


def test_bbbc038_training_sample_labels_instances_in_mask_filename_order(dataset_path):
    dataset = BBBC038Dataset(is_training=True, path=dataset_path, img_size=8)

    _, _, _, instance_map = dataset[0]

    assert instance_map.dtype == torch.int64
    assert set(instance_map.unique().tolist()) == {0, 1, 2}
    assert instance_map[1, 1] == 1
    assert instance_map[6, 6] == 2
    assert instance_map[1, 6] == 0


def test_bbbc038_test_sample_returns_no_mask_and_writes_no_stdout(dataset_path, capsys):
    dataset = BBBC038Dataset(is_training=False, path=dataset_path, img_size=8)

    sample_id, image = dataset[1]

    assert sample_id == "sample-b"
    assert image.shape == (3, 8, 8)
    assert capsys.readouterr().out == ""
