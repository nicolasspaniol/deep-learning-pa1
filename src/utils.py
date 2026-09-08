import torch
import numpy as np
import cv2

def is_binary(mask):
    return torch.all((mask == 0) | (mask == 1))


def iou_score(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape, 'tensors must have the same shape'
    assert is_binary(true_mask), 'tensors must contain only 0/1s'
    assert is_binary(pred_mask), 'tensors must contain only 0/1s'

    intersection = (true_mask * pred_mask).sum()
    union = ((true_mask + pred_mask) > 0).sum()
    return intersection / union


def dice_score(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape, 'tensors must have the same shape'
    assert is_binary(true_mask), 'tensors must contain only 0/1s'
    assert is_binary(pred_mask), 'tensors must contain only 0/1s'

    intersection = (true_mask * pred_mask).sum()
    total = (true_mask + pred_mask).sum()
    return 2 * intersection / total

def probability_to_instances(probability, threshold=0.5):

    binary_mask = (probability >= threshold).astype(np.uint8)

    number_of_labels, instance_map = cv2.connectedComponents(
        binary_mask,
        connectivity=8
    )

    return instance_map.astype(np.int32)

def get_instance_ids(instance_map):
    ids = np.unique(instance_map)
    return ids[ids != 0]


def instance_iou_matrix(gt_map, pred_map):
    if torch.is_tensor(gt_map):
        gt_map = gt_map.detach().cpu().numpy()

    if torch.is_tensor(pred_map):
        pred_map = pred_map.detach().cpu().numpy()

    gt_ids = get_instance_ids(gt_map)
    pred_ids = get_instance_ids(pred_map)

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for i, gt_id in enumerate(gt_ids):
        gt_mask = torch.from_numpy((gt_map == gt_id).astype(np.float32))

        for j, pred_id in enumerate(pred_ids):
            pred_mask = torch.from_numpy((pred_map == pred_id).astype(np.float32))

            iou = iou_score(gt_mask, pred_mask)

            iou_matrix[i, j] = iou.item()

    return iou_matrix



if __name__ == '__main__':
    a = torch.tensor([1, 0, 1, 1])
    b = torch.tensor([1, 0, 0, 1])

    assert is_binary(a)
    assert not is_binary(torch.tensor([0, 2, 1]))

    assert torch.isclose(iou_score(a, a), torch.tensor(1.0))
    assert torch.isclose(iou_score(a, b), torch.tensor(2/3))

    assert torch.isclose(dice_score(a, a), torch.tensor(1.0))
    assert torch.isclose(dice_score(a, b), torch.tensor(0.8))

    c = np.array([[0, 1, 1], 
                  [0, 2, 0]])

    ids = get_instance_ids(c)
    assert np.array_equal(ids, np.array([1, 2]))

    gt_map = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 2]
    ])

    pred_map = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 2]
    ])

    expected_matrix = np.array([
        [0.75, 0.0],  
        [0.0,  1.0]   
    ], dtype=np.float32)

    iou_matrix_result = instance_iou_matrix(gt_map, pred_map)
    assert np.allclose(iou_matrix_result, expected_matrix, atol=1e-5)