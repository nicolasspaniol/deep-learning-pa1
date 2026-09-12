# local files
import utils
from synthetic_dataset import SyntheticEllipseDataset
from resunet import ResUNet
from bbbc038_dataset import BBBC038Dataset
from center_offset_dataset import CenterOffsetDataset
from plotting import plot_sample, plot_prediction

# libraries
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)

# carrega o dataset
dataset = CenterOffsetDataset(BBBC038Dataset(True, './data/stage1_train'))
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
test_dataset = BBBC038Dataset(False, './data/stage1_test')
