# local files
from src import utils
from src.synthetic_dataset import SyntheticEllipseDataset
from src.resunet import ResUNet
from src.bbbc038_dataset import BBBC038Dataset
from src.center_offset_dataset import CenterOffsetDataset
from src.plotting import plot_sample, plot_prediction
from src.device import device
from src.datasets import bbbc038_center_offsets

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


# carrega o dataset
train_ds, val_ds, test_ds = bbbc038_center_offsets()
test_dataset = BBBC038Dataset(False, './data/stage1_test')
