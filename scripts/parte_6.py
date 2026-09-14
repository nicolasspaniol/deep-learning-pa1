# local files
from src.bbbc038_dataset import BBBC038Dataset
from src.corrupted_dataset import CorruptedBBBC038Dataset
from src.resunet import ResUNet
from src import utils
from src.device import device
from src.datasets import bbbc038

# libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sys import argv

# torch
import torch


CORRUPTIONS = ['blur', 'noise', 'brightness_contrast']
INTENSITIES = [0, 1, 2, 3]  # 0 = uncorrupted baseline


@torch.no_grad()
def run_inference(model, image, device, heatmap_threshold=0.3, mask_threshold=0.3):
    """
    image: (3, H, W) tensor -> predicted instance map (H, W) int64 tensor.
    Returns None if no peaks are detected (nothing to assign instances to).
    """
    image_batch = image.unsqueeze(0).to(device)
    pred = model(image_batch).cpu()[0]  # (4, H, W)

    pred_heatmap = pred[0].numpy()
    pred_dx, pred_dy = pred[1].numpy(), pred[2].numpy()
    pred_foreground = torch.sigmoid(pred[3]).numpy()
    pred_mask = pred_foreground > mask_threshold

    xs, ys = utils.find_peaks(pred_heatmap, threshold=heatmap_threshold)
    if len(xs) == 0:
        return torch.zeros(pred_heatmap.shape, dtype=torch.int64)

    centroids = np.stack([xs, ys], axis=1)  # (N, 2), (x, y)
    positions = torch.stack([
        torch.arange(pred_heatmap.shape[1]).repeat(pred_heatmap.shape[0], 1).float(),  # x grid
        torch.arange(pred_heatmap.shape[0]).repeat(pred_heatmap.shape[1], 1).T.float(),  # y grid
    ])

    background_mask = torch.from_numpy(~pred_mask)
    instance_map = utils.assign_instances(positions, centroids, background_mask)

    return instance_map


def evaluate_dataset(model, dataset, device):
    """Runs inference over `dataset` and returns compute_map + count MAE."""
    gt_pred_pairs = []

    for index in tqdm(range(len(dataset)), leave=False):
        _, image, _, gt_instance_map = dataset[index]
        pred_instance_map = run_inference(model, image, device)
        gt_pred_pairs.append((gt_instance_map.numpy(), pred_instance_map.numpy()))

    map_result = utils.compute_map(gt_pred_pairs)
    count_mae = utils.compute_count_mae(gt_pred_pairs)

    return map_result['final_map'], count_mae


def main(weights_path: str, data_path: str = './data/stage1_train'):
    print('Carregando o dataset de teste...')
    _, _, test_ds = bbbc038()

    print('Carregando o modelo já treinado...')
    model = ResUNet(3, 4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    print('Rodando o baseline (sem corrupção)...')
    baseline_map, baseline_mae = evaluate_dataset(model, test_ds, device)
    print(f'  mAP: {baseline_map:.4f} | Count MAE: {baseline_mae:.4f}')

    results = {corruption: {0: baseline_map} for corruption in CORRUPTIONS}

    for corruption in CORRUPTIONS:
        for intensity in [1, 2, 3]:
            print(f'Avaliando corrupção "{corruption}" na intensidade {intensity}...')
            corrupted_ds = CorruptedBBBC038Dataset(test_ds, corruption, intensity)
            map_score, mae_score = evaluate_dataset(model, corrupted_ds, device)
            print(f'  mAP: {map_score:.4f} | Count MAE: {mae_score:.4f}')
            results[corruption][intensity] = map_score

    print('Plotando a curva de degradação...')
    plt.figure(figsize=(7, 5))
    for corruption in CORRUPTIONS:
        scores = [results[corruption][i] for i in INTENSITIES]
        plt.plot(INTENSITIES, scores, marker='o', label=corruption)

    plt.xlabel('Intensidade da corrupção (0 = sem corrupção)')
    plt.ylabel('mAP')
    plt.title('Curva de degradação do mAP por tipo de corrupção')
    plt.xticks(INTENSITIES)
    plt.legend()
    plt.tight_layout()
    plt.savefig('degradation_curve.png')
    plt.show()


if __name__ == '__main__':
    weights_path = argv[1]
    data_path = argv[2] if len(argv) > 2 else './data/stage1_train'
    main(weights_path, data_path)
