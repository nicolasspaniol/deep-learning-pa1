import torch


def is_binary(mask):
    return torch.all((mask == 0) | (mask == 1))


def iou_score(true_mask, pred_mask):
    assert is_binary(true_mask)
    assert is_binary(pred_mask)

    intersection = (true_mask * pred_mask).sum()
    union = ((true_mask + pred_mask) > 0).sum()
    return intersection / union


def dice_score(true_mask, pred_mask):
    assert is_binary(true_mask)
    assert is_binary(pred_mask)

    intersection = (true_mask * pred_mask).sum()
    total = (true_mask + pred_mask).sum()
    return 2 * intersection / total


if __name__ == '__main__':
    a = torch.tensor([1, 0, 1, 1])
    b = torch.tensor([1, 0, 0, 1])

    assert is_binary(a)
    assert not is_binary(torch.tensor([0, 2, 1]))

    assert torch.isclose(iou_score(a, a), torch.tensor(1.0))
    assert torch.isclose(iou_score(a, b), torch.tensor(2/3))

    assert torch.isclose(dice_score(a, a), torch.tensor(1.0))
    assert torch.isclose(dice_score(a, b), torch.tensor(0.8))
