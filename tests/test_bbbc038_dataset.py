import cv2
import numpy as np
import pytest
import torch

from bbbc038_dataset import BBBC038Dataset, InstanceEvaluationSubset
from utils import is_binary


@pytest.fixture
def bbbc038_root(tmp_path):
    sample_id = "sample-1"
    sample_dir = tmp_path / "stage1_train" / sample_id
    images_dir = sample_dir / "images"
    masks_dir = sample_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    image = np.full((4, 4, 4), [64, 128, 255, 17], dtype=np.uint8)
    first_mask = np.zeros((4, 4), dtype=np.uint8)
    second_mask = np.zeros((4, 4), dtype=np.uint8)
    first_mask[:2, :2] = 255
    second_mask[2:, 2:] = 255
    assert cv2.imwrite(str(images_dir / f"{sample_id}.png"), image), "test fixture image must be written"
    assert cv2.imwrite(str(masks_dir / "first.png"), first_mask), "first test fixture mask must be written"
    assert cv2.imwrite(str(masks_dir / "second.png"), second_mask), "second test fixture mask must be written"
    return tmp_path


def test_bbbc038_dataset_requires_existing_data(tmp_path):
    with pytest.raises(RuntimeError, match="Dataset not found"):
        BBBC038Dataset(root_dir=tmp_path)


def test_bbbc038_dataset_loads_resizes_and_merges_masks(bbbc038_root):
    dataset = BBBC038Dataset(root_dir=bbbc038_root, img_size=8)
    image_tensor, mask_tensor = dataset[0]

    assert len(dataset) == 1, "dataset length must equal the number of sample directories"
    assert dataset.ids == ["sample-1"], "dataset IDs must be sorted sample directory names"
    assert image_tensor.shape == (3, 8, 8), "image must be RGB CHW and resized to img_size"
    assert mask_tensor.shape == (8, 8), "semantic mask must be 2D and resized to img_size"
    assert image_tensor.dtype == torch.float32, "image must use float32"
    assert mask_tensor.dtype == torch.float32, "mask must use float32"
    assert image_tensor.min() >= 0 and image_tensor.max() <= 1, "image values must be normalized to [0, 1]"
    assert is_binary(mask_tensor), "merged semantic mask must contain only 0 and 1"
    assert mask_tensor[1, 1] == 1 and mask_tensor[6, 6] == 1, "merged mask must retain foreground from both instance masks"
    assert mask_tensor[1, 6] == 0, "merged mask must retain background outside all instance masks"


def test_instance_evaluation_subset_returns_selected_sample_and_instance_labels(bbbc038_root):
    dataset = BBBC038Dataset(root_dir=bbbc038_root, img_size=8)
    subset = InstanceEvaluationSubset(dataset, indices=[0], img_size=8)

    image, semantic_mask, instance_map, image_id = subset[0]

    assert len(subset) == 1, "subset length must equal the number of selected indices"
    assert image_id == "sample-1", "subset must return the selected sample ID"
    assert image.shape == (3, 8, 8), "subset must return the original image tensor"
    assert torch.equal(semantic_mask, dataset[0][1]), "subset semantic mask must match the original dataset mask"
    assert instance_map.dtype == torch.int64, "instance map must use integer labels"
    assert set(instance_map.unique().tolist()) == {0, 1, 2}, "instance map must label background and both masks separately"
    assert instance_map[1, 1] == 1 and instance_map[6, 6] == 2, "instance labels must follow sorted mask-file order"
