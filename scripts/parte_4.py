# local files
from src import utils
from src.resunet import ResUNet
from src.plotting import plot_prediction
from src.device import device
from src.datasets import bbbc038_center_offsets_tiled
from src.utils import probability_to_instances

# libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms.functional import resize, InterpolationMode

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
    n_tiles = test_ds.n_tiles
    start = sample_idx * n_tiles ** 2

    fig, axes = plt.subplots(n_tiles, n_tiles, figsize=(9, 9))

    for tile_idx in range(n_tiles ** 2):
        tile_id, image, semantic_mask, instance_map, _ = test_ds[start + tile_idx]
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
        pred = model(image_batch).cpu()[0]  # (4, H, W)

        pred_heatmap = pred[0].numpy()
        pred_dx, pred_dy = pred[1].numpy(), pred[2].numpy()
        pred_foreground = torch.sigmoid(pred[3]).numpy()
        pred_mask = (pred_foreground > mask_threshold).astype(np.float32)

        return pred_heatmap, pred_dx, pred_dy, pred_mask

    fig, axes = plt.subplots(n_tiles, n_tiles, figsize=(9, 9))
    for i in range(n_tiles ** 2):
        sample = test_ds[start + i]
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

        tile_id, image, semantic_mask, instance_map, _ = test_ds[start + i]
        ax = axes[i // n_tiles, i % n_tiles]
        ax.imshow(instances, cmap='inferno')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    def _resize_nearest(tensor, size):
        t = tensor.unsqueeze(0).unsqueeze(0).float()
        t = resize(t, size, interpolation=InterpolationMode.NEAREST)
        return t[0, 0].long()

    def merge_tile_instances(tile_instance_maps, tile_bounds, full_shape, overlap_frac_thresh=0.3):
        H, W = full_shape
        canvas = torch.zeros((H, W), dtype=torch.int64)
        next_id = 1

        for local_map, (y0, y1, x0, x1) in zip(tile_instance_maps, tile_bounds):
            canvas_region = canvas[y0:y1, x0:x1]

            for lid in torch.unique(local_map).tolist():
                if lid == 0:
                    continue
                lmask = local_map == lid
                overlap_vals = canvas_region[lmask]
                overlap_vals = overlap_vals[overlap_vals > 0]

                if overlap_vals.numel() > 0:
                    gid = int(overlap_vals.mode().values)
                    frac = (overlap_vals == gid).sum().item() / lmask.sum().item()
                    if frac >= overlap_frac_thresh:
                        canvas_region[lmask] = gid
                        continue

                canvas_region[lmask] = next_id
                next_id += 1

        return canvas

    def merge_tile_instances_naive(tile_instance_maps, tile_bounds, full_shape):
        """Stitches per-tile instance maps into one full-size map with no
        cross-tile reconciliation -- every local instance gets a fresh global
        ID, so objects detected in more than one overlapping tile appear as
        separate (duplicated) instances."""
        H, W = full_shape
        canvas = torch.zeros((H, W), dtype=torch.int64)
        next_id = 1

        for local_map, (y0, y1, x0, x1) in zip(tile_instance_maps, tile_bounds):
            canvas_region = canvas[y0:y1, x0:x1]

            for lid in torch.unique(local_map).tolist():
                if lid == 0:
                    continue
                lmask = local_map == lid
                canvas_region[lmask] = next_id
                next_id += 1

        return canvas

    print('Fazendo o merge das instâncias nos tiles...')

    base_untiled = test_ds.base_dataset.base_dataset
    _, full_image, _, _ = base_untiled[sample_idx]
    h, w = full_image.shape[1:]
    tile_bounds = [test_ds._tile_bounds(i) for i in range(n_tiles ** 2)]

    tile_instance_maps = []
    for tile_idx in range(n_tiles ** 2):
        sample = test_ds[start + tile_idx]
        pred_heatmap, pred_dx, pred_dy, pred_mask = run_inference(model, sample, device)
        tile_instance_maps.append(torch.from_numpy(probability_to_instances(pred_mask)).long())

    merged_instances_naive = merge_tile_instances_naive(tile_instance_maps, tile_bounds, (h, w))

    plt.figure(figsize=(6, 6))
    plt.imshow(merged_instances_naive, cmap='inferno')
    plt.title(f'Naive merge ({merged_instances_naive.max().item()} objects, duplicates kept)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    merged_instances = merge_tile_instances(tile_instance_maps, tile_bounds, (h, w))

    plt.figure(figsize=(6, 6))
    plt.imshow(merged_instances, cmap='inferno')
    plt.title(f'Merged instances ({merged_instances.max().item()} objects)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
