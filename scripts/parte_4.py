# local files
import src.utils
from src.resunet import ResUNet
from src.bbbc038_dataset import BBBC038Dataset
from src.center_offset_dataset import CenterOffsetDataset
from src.tiled_dataset import TiledBBBC038Dataset
from src.plotting import plot_prediction

# libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device.type)

print('Carregando o dataset...')
dataset = TiledBBBC038Dataset(BBBC038Dataset(True, './data/stage1_train'), n_tiles=3, overlap=.1)

print('Visualizando os tiles de uma amostra do dataset...')

sample_idx = 7
n_tiles = dataset.n_tiles
start = sample_idx * n_tiles ** 2

fig, axes = plt.subplots(n_tiles, n_tiles, figsize=(9, 9))

for tile_idx in range(n_tiles ** 2):
    tile_id, image, semantic_mask, instance_map = dataset[start + tile_idx]
    ax = axes[tile_idx // n_tiles, tile_idx % n_tiles]
    ax.imshow(image.permute(1, 2, 0))
    ax.axis('off')

plt.tight_layout()
plt.show()

print('Carregando o modelo já treinado...')

model = ResUNet(3, 3).to(device)
model.load_state_dict(torch.load('weights/instance_weights', map_location=device, weights_only=True))
model.eval()

print('Fazendo a inferência nos recortes da imagem...')

dataset = CenterOffsetDataset(dataset)

@torch.no_grad()
def run_inference(model, sample, device, mask_threshold=0.3):
    _, image, _, _, _ = sample

    image_batch = image.unsqueeze(0).to(device)
    pred = model(image_batch).cpu()[0]  # (3, H, W)

    pred_heatmap = pred[0].numpy()
    pred_dx, pred_dy = pred[1].numpy(), pred[2].numpy()
    pred_mask = (pred_heatmap > mask_threshold).astype(np.float32)

    return pred_heatmap, pred_dx, pred_dy, pred_mask

for i in range(8):
    sample = dataset[start + i]
    plot_prediction(sample, *run_inference(model, sample, device))
    plt.show()
