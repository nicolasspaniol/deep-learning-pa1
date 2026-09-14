from src.datasets import bbbc038_center_offsets
from src.plotting import plot_sample
from src.resunet import ResUNet

from sys import argv
import random
import datetime
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def main(epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device.type)

    # dataset -----------------------------
    train_ds, val_ds, test_ds = bbbc038_center_offsets()

    # model -------------------------------
    model = ResUNet(3, 4).to(device)

    # training process -------------------------
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    heatmap_loss_fn = nn.MSELoss()
    offset_loss_fn = nn.L1Loss(reduction='none')  # 'none' pra poder mascarar pixel a pixel
    foreground_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    offset_loss_weight = 1
    foreground_loss_weight = 1

    model.train()
    for epoch in (pbar := tqdm(range(epochs))):
        for batch, (_, image, heatmap, offsets, offset_mask) in enumerate(loader):
            image = image.to(device)
            heatmap = heatmap.to(device)
            offsets = offsets.to(device)
            offset_mask = offset_mask.to(device)

            pred = model(image)
            pred_heatmap =    pred[:, 0:1]
            pred_offsets =    pred[:, 1:3]
            pred_foreground = pred[:, 3:4]

            loss_heatmap = heatmap_loss_fn(pred_heatmap, heatmap)

            raw_offset_loss = offset_loss_fn(pred_offsets, offsets)
            mask_2ch = offset_mask.expand_as(raw_offset_loss)
            loss_offsets = (raw_offset_loss * mask_2ch).sum() / mask_2ch.sum().clamp(min=1.0)

            loss_foreground = foreground_loss_fn(pred_foreground, offset_mask)

            loss = loss_heatmap + offset_loss_weight * loss_offsets + foreground_loss_weight * loss_foreground
            pbar.set_description(f"Loss {float(loss.item()):.2f}")
            pbar.refresh()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # saving weights ------------------------------------
    weights_filename = 'weights/weights_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), weights_filename)
    print(f'weights saved at "{weights_filename}"')


if __name__ == '__main__':
    epochs = int(argv[1])
    main(epochs)
