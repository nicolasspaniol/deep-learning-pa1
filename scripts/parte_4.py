# local files
from src import utils
from src.resunet import ResUNet
from src.plotting import plot_prediction
from src.device import device
from src.datasets import bbbc038_center_offsets_tiled

# libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def main():
    print('Carregando o dataset...')
    train_ds, val_ds, test_ds = bbbc038_center_offsets_tiled()

    print('Visualizando os tiles de uma amostra do dataset...')

    sample_idx = 7
    n_tiles = 4
    start = sample_idx * n_tiles ** 2

    fig, axes = plt.subplots(n_tiles, n_tiles, figsize=(9, 9))

    for tile_idx in range(n_tiles ** 2):
        tile_id, image, semantic_mask, instance_map, _ = train_ds[start + tile_idx]
        ax = axes[tile_idx // n_tiles, tile_idx % n_tiles]
        ax.imshow(image.permute(1, 2, 0))
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print('Carregando o modelo já treinado...')

    model = ResUNet(3, 4).to(device)
    model.load_state_dict(torch.load('weights/weights_20260913_190326', map_location=device, weights_only=True))
    model.eval()

    print('Fazendo a inferência nos recortes da imagem...')

    @torch.no_grad()
    def run_inference(model, sample, device, mask_threshold=0.3):
        _, image, _, _, _ = sample

        image_batch = image.unsqueeze(0).to(device)
        pred = model(image_batch).cpu()[0]  # (3, H, W)

        pred_heatmap = pred[0].numpy()
        pred_dx, pred_dy = pred[1].numpy(), pred[2].numpy()
        pred_mask = (pred_heatmap > mask_threshold).astype(np.float32)

        return pred_heatmap, pred_dx, pred_dy, pred_mask

    fig, axes = plt.subplots(n_tiles, n_tiles, figsize=(9, 9))
    for i in range(n_tiles ** 2):
        sample = train_ds[start + i]
        _, image, _, _, _ = sample
        image_batch = image.unsqueeze(0).to(device)

        pred = model(image_batch).cpu()[0].detach()

        xs, ys = utils.find_peaks(pred[0].numpy(), threshold=0.3, nms_kernel=5, top_k=50)
        centroids = np.stack([ys, xs], axis=1)  # (N, 2) as (row, col) to match positions

        positions = torch.stack(torch.meshgrid(
            torch.arange(pred.shape[1], dtype=torch.float32),
            torch.arange(pred.shape[2], dtype=torch.float32),
            indexing='ij'
        ))
        positions[0] += pred[2]
        positions[1] += pred[1]
        bg_mask = pred[3] < .5
        instances = utils.assign_instances(positions, centroids, background_mask=bg_mask)

        tile_id, image, semantic_mask, instance_map, _ = train_ds[start + i]
        ax = axes[i // n_tiles, i % n_tiles]
        ax.imshow(instances, cmap='inferno')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
