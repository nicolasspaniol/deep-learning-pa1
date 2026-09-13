from src.plotting import plot_sample, plot_prediction
from src.datasets import  bbbc038_center_offsets
from src.resunet import ResUNet
from src import utils
from src.device import device

import matplotlib.pyplot as plt
import random
from sys import argv
import torch
import numpy as np


def main(weights_path: str):
    train_ds, val_ds, test_ds = bbbc038_center_offsets()

    model = ResUNet(3, 4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    @torch.no_grad()
    def run_inference(model, sample, device, mask_threshold=0.3):
        """
        Run the model on a single sample and return plain numpy arrays ready
        for plotting: (pred_heatmap, pred_dx, pred_dy, pred_mask).
        """
        model.eval()
        _, image, _, _, _ = sample

        image_batch = image.unsqueeze(0).to(device)
        pred = model(image_batch).cpu()[0]  # (4, H, W)

        pred_heatmap = pred[0].numpy()
        pred_dx, pred_dy = pred[1].numpy(), pred[2].numpy()
        pred_foreground = torch.sigmoid(pred[3]).numpy()
        pred_mask = (pred_foreground > mask_threshold).astype(np.float32)

        return pred_heatmap, pred_dx, pred_dy, pred_mask

    while True:
        sample = test_ds[random.randint(0, 50)]
        pred_heatmap, pred_dx, pred_dy, pred_mask = run_inference(model, sample, device)
        peaks = utils.find_peaks(pred_heatmap)

        plot_prediction(sample, pred_heatmap, pred_dx, pred_dy, pred_mask, peaks=peaks)
        plt.show()


if __name__ == '__main__':
    weights_path = argv[1]
    main(weights_path)

