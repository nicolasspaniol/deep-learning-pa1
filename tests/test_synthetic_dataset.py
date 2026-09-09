import torch

from synthetic_dataset import SyntheticEllipseDataset


def test_synthetic_ellipse_dataset_length_matches_requested_sample_count():
    dataset = SyntheticEllipseDataset(num_samples=7)

    assert len(dataset) == 7, "dataset length must equal num_samples"


def test_synthetic_ellipse_dataset_returns_normalized_float_tensors():
    dataset = SyntheticEllipseDataset(num_samples=1, img_size=32)

    image, mask = dataset[0]

    assert image.shape == (3, 32, 32), "image must be a 3-channel CHW tensor at img_size"
    assert mask.shape == (32, 32), "mask must be a 2D tensor at img_size"
    assert image.dtype == torch.float32, "image must use float32"
    assert mask.dtype == torch.float32, "mask must use float32"
    assert image.min() >= 0 and image.max() <= 1, "image values must be normalized to [0, 1]"
    assert mask.min() >= 0 and mask.max() <= 1, "mask values must be normalized to [0, 1]"
    assert torch.all((mask == 0) | (mask == 1)), "mask must contain only background and foreground values"
