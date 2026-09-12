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


def greedy_matching(iou_matrix, threshold):
    n_gt, n_pred = iou_matrix.shape

    gt_candidates, pred_candidates = np.where(iou_matrix >= threshold)

    candidate_ious = iou_matrix[gt_candidates, pred_candidates]

    # Pares con IoU maior primero
    order = np.argsort(candidate_ious)[::-1]

    used_gt = set()
    used_pred = set()

    matches = []

    for index in order:
        gt_index = int(gt_candidates[index])
        pred_index = int(pred_candidates[index])

        if gt_index in used_gt:
            continue

        if pred_index in used_pred:
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)

        matches.append((gt_index, pred_index, float(iou_matrix[gt_index, pred_index])))

    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    return tp, fp, fn


def evaluate_instance_prediction(gt_map, pred_map):
    iou_matrix = instance_iou_matrix(gt_map, pred_map)

    results = []

    IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)

    for threshold in IOU_THRESHOLDS:
        tp, fp, fn = greedy_matching(iou_matrix, threshold)

        denominator = tp + fp + fn

        if denominator == 0:
            ap = 1.0
        else:
            ap = tp / denominator

        results.append({
            "iou_threshold": float(threshold),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "AP": ap
        })

    return results


def compute_map(model, dataset, utils, device, threshold):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    image_maps = []
    count_errors = []
    ap_by_thr = {}

    with torch.no_grad():
        for image_id, image, semantic_mask, gt_batch in loader:
            probability = torch.sigmoid(model(image.to(device)))[0, 0].cpu().numpy()
            pred_map = probability_to_instances(probability, threshold=threshold)
            gt_map = gt_batch[0].numpy()

            results = evaluate_instance_prediction(gt_map, pred_map)

            image_maps.append(np.mean([r["AP"] for r in results]))
            count_errors.append(abs(
                len(utils.get_instance_ids(pred_map)) - len(utils.get_instance_ids(gt_map))
            ))

            for r in results:
                ap_by_thr.setdefault(r["iou_threshold"], []).append(r["AP"])

    return {
        "final_map": float(np.mean(image_maps)),
        "count_mae": float(np.mean(count_errors)),
        "ap_by_threshold": {thr: float(np.mean(aps)) for thr, aps in sorted(ap_by_thr.items())},
    }
