import torch

from synthetic_dataset import SyntheticEllipseDataset


def test_synthetic_ellipse_dataset_length_matches_requested_sample_count():
    dataset = SyntheticEllipseDataset(num_samples=7)

    assert len(dataset) == 7, "dataset length must equal num_samples"


def test_synthetic_ellipse_dataset_returns_normalized_tensors_and_instance_map():
    dataset = SyntheticEllipseDataset(num_samples=1, img_size=32)

    image, semantic_mask, instance_map = dataset[0]

    assert image.shape == (3, 32, 32), "image must be a 3-channel CHW tensor at img_size"
    assert semantic_mask.shape == (32, 32), "semantic mask must be a 2D tensor at img_size"
    assert instance_map.shape == (32, 32), "instance map must be a 2D tensor at img_size"
    assert image.dtype == torch.float32, "image must use float32"
    assert semantic_mask.dtype == torch.float32, "semantic mask must use float32"
    assert instance_map.dtype == torch.int64, "instance map must use int64 labels"
    assert image.min() >= 0 and image.max() <= 1, "image values must be normalized to [0, 1]"
    assert semantic_mask.min() >= 0 and semantic_mask.max() <= 1, "semantic mask values must be normalized to [0, 1]"
    assert torch.all((semantic_mask == 0) | (semantic_mask == 1)), "semantic mask must contain only background and foreground values"
    assert torch.equal(semantic_mask, (instance_map > 0).float())


def test_synthetic_ellipse_dataset_assigns_distinct_ids_to_non_overlapping_ellipses(monkeypatch):
    # One call determines the ellipse count, followed by six calls per ellipse:
    # center x/y, axes x/y, angle, and grayscale value.
    values = iter([2, 10, 10, 5, 5, 0, 50, 22, 22, 5, 5, 0, 100])
    monkeypatch.setattr("synthetic_dataset.random.randint", lambda *_: next(values))
    dataset = SyntheticEllipseDataset(num_samples=1, img_size=32)

    _, semantic_mask, instance_map = dataset[0]

    assert set(instance_map.unique().tolist()) == {0, 1, 2}
    assert instance_map[10, 10] == 1
    assert instance_map[22, 22] == 2
    assert torch.equal(semantic_mask, (instance_map > 0).float())
