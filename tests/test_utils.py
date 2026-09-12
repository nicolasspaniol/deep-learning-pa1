import numpy as np
import pytest
import torch

from utils import *


def test_is_binary_accepts_only_zero_and_one_values():
    assert is_binary(torch.tensor([0, 1, 1])), "a mask containing only 0 and 1 must be binary"
    assert is_binary(torch.zeros(2, 2)), "an all-background mask must be binary"
    assert not is_binary(torch.tensor([0, 0.5, 1])), "fractional mask values must not be binary"
    assert not is_binary(torch.tensor([0, 2, 1])), "values above 1 must not be binary"


def test_iou_and_dice_scores_match_expected_overlap():
    first = torch.tensor([1, 0, 1, 1])
    second = torch.tensor([1, 0, 0, 1])

    assert torch.isclose(iou_score(first, first), torch.tensor(1.0)), "identical masks must have IoU 1"
    assert torch.isclose(iou_score(first, second), torch.tensor(2 / 3)), "IoU must be intersection divided by union"
    assert torch.isclose(dice_score(first, first), torch.tensor(1.0)), "identical masks must have Dice 1"
    assert torch.isclose(dice_score(first, second), torch.tensor(0.8)), "Dice must use twice the intersection"


def test_binary_scores_reject_invalid_inputs():
    with pytest.raises(AssertionError, match="same shape"):
        iou_score(torch.tensor([1]), torch.tensor([[1]]))
    with pytest.raises(AssertionError, match="only 0/1s"):
        dice_score(torch.tensor([1, 2]), torch.tensor([1, 1]))


def test_probability_to_instances_uses_threshold_and_eight_connectivity():
    probability = np.array([[0.5, 0.1], [0.1, 0.8]], dtype=np.float32)

    instance_map = probability_to_instances(probability)

    assert instance_map.dtype == np.int32, "instance map must use int32 labels"
    assert instance_map[0, 0] == instance_map[1, 1] == 1, "diagonally touching foreground pixels must form one instance"
    assert instance_map[0, 1] == instance_map[1, 0] == 0, "values below threshold must remain background"


def test_get_instance_ids_excludes_background_and_sorts_ids():
    instance_map = np.array([[3, 0, 1], [0, 2, 3]])

    assert np.array_equal(get_instance_ids(instance_map), np.array([1, 2, 3])), "instance IDs must exclude zero and be sorted"


def test_instance_iou_matrix_supports_numpy_and_torch_inputs():
    ground_truth = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]])
    prediction = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]])
    expected = np.array([[0.75, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = instance_iou_matrix(torch.from_numpy(ground_truth), torch.from_numpy(prediction))

    assert result.shape == (2, 2), "IoU matrix dimensions must match ground-truth and prediction instance counts"
    assert result.dtype == np.float32, "IoU matrix must use float32"
    assert np.allclose(result, expected, atol=1e-5), "IoU matrix values must match pairwise instance overlaps"


def test_greedy_matching_counts_true_false_positives_and_false_negatives():
    iou_matrix = np.array([[0.9, 0.6], [0.8, 0.1]], dtype=np.float32)

    true_positives, false_positives, false_negatives = greedy_matching(iou_matrix, threshold=0.5)

    assert true_positives == 1, "the highest-IoU match must prevent reuse of its row and column"
    assert false_positives == 1, "the unmatched predicted instance must count as a false positive"
    assert false_negatives == 1, "the unmatched ground-truth instance must count as a false negative"


def test_evaluate_instance_prediction_reports_all_thresholds_and_empty_case():
    results = evaluate_instance_prediction(np.zeros((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32))

    assert len(results) == 10, "evaluation must report thresholds from 0.50 through 0.95"
    assert results[0]["iou_threshold"] == pytest.approx(0.5), "first evaluation threshold must be 0.50"
    assert results[-1]["iou_threshold"] == pytest.approx(0.95), "last evaluation threshold must be 0.95"
    assert all(result["TP"] == result["FP"] == result["FN"] == 0 for result in results), "empty maps must have no detections"
    assert all(result["AP"] == 1.0 for result in results), "two empty maps must receive perfect AP"
