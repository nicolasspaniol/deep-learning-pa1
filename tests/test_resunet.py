from resunet import ResUNet

def test_resunet():
    from synthetic_dataset import SyntheticEllipseDataset
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model = ResUNet(3, 1)
    model.eval()

    dataset = SyntheticEllipseDataset(num_samples=1)
    dataloader = DataLoader(dataset)

    img, mask = next(iter(dataloader))
    y = F.sigmoid(model(img))

    assert y.shape[0] == img.shape[0], 'output size is the same as input'
    # shape[1] é o número de canais, que pode variar
    assert y.shape[2] == img.shape[2] and y.shape[3] == img.shape[3], 'output image resolution is the same as input'
