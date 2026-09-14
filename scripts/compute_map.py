# local files
from src import utils
from src.resunet import ResUNet
from src.device import device
from src.datasets import bbbc038_center_offsets

# libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch
import torch


def main():
    print('Carregando o dataset...')
    train_ds, val_ds, test_ds = bbbc038_center_offsets()

    print('Carregando o modelo já treinado...')

    model = ResUNet(3, 4).to(device)
    model.load_state_dict(torch.load('weights/weights_20260913_190326', map_location=device, weights_only=True))
    model.eval()

    print('Fazendo a inferência no conjunto de teste...')

    @torch.no_grad()
    def run_inference(model, sample, device, mask_threshold=0.3):
        _, image, _, _, _ = sample

        image_batch = image.unsqueeze(0).to(device)
        pred = model(image_batch).cpu()[0]  # (4, H, W)

        return pred

    gt_pred_pairs = []

    for idx in tqdm(range(len(test_ds))):
        sample = test_ds[idx]
        _, _, _, _, _ = sample
        pred = run_inference(model, sample, device)

        xs, ys = utils.find_peaks(pred[0].numpy(), threshold=0.3, nms_kernel=5, top_k=50)

        if len(xs) == 0:
            # no peaks found -- everything predicted as background
            H, W = pred.shape[1:]
            pred_instances = torch.zeros((H, W), dtype=torch.int64)
        else:
            centroids = np.stack([ys, xs], axis=1)  # (N, 2) as (row, col)

            positions = torch.stack(torch.meshgrid(
                torch.arange(pred.shape[1], dtype=torch.float32),
                torch.arange(pred.shape[2], dtype=torch.float32),
                indexing='ij'
            ))
            positions[0] += pred[2]
            positions[1] += pred[1]
            bg_mask = pred[3] < .5
            pred_instances = utils.assign_instances(positions, centroids, background_mask=bg_mask)

        # ground-truth instance map: recover it from the base (untiled) dataset
        base_untiled = test_ds.dataset.base_dataset
        real_idx = test_ds.indices[idx] if hasattr(test_ds, 'indices') else idx
        _, _, _, gt_instance = base_untiled[real_idx]

        gt_pred_pairs.append((gt_instance, pred_instances))

    print('Calculando mAP...')

    map_result = utils.compute_map(gt_pred_pairs)

    print(f"mAP final (sem tiling): {map_result['final_map']:.4f}")

    thresholds = sorted(map_result['ap_by_threshold'])
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, [map_result['ap_by_threshold'][t] for t in thresholds], marker='o')
    plt.xlabel('IoU threshold')
    plt.ylabel('AP')
    plt.title('AP by IoU threshold (baseline, no tiling)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
