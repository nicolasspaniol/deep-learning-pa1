import torch

from resunet import ResUNet


def test_resunet_preserves_batch_size_and_spatial_resolution():
    model = ResUNet(3, 1).eval()
    image = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        output = model(image)

    assert output.shape[0] == image.shape[0], "output batch size must match input batch size"
    assert output.shape[2:] == image.shape[2:], "output height and width must match the input"


def test_resunet_uses_requested_output_channels_and_returns_finite_logits():
    model = ResUNet(3, 2).eval()
    image = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        output = model(image)

    assert output.shape[1] == 2, "output channel count must equal out_channels"
    assert torch.isfinite(output).all(), "model logits must not contain NaN or infinity"
