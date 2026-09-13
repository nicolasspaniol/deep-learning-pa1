from torch import Generator
from torch.utils.data import random_split

from bbbc038_dataset import BBBC038Dataset
from center_offset_dataset import CenterOffsetDataset

_gen = Generator().manual_seed(42)
_img_size = 128


def bbbc038_center_offsets():
    dataset = CenterOffsetDataset(BBBC038Dataset(True, './data/stage1_train', img_size=_img_size))
    return random_split(dataset, [0.8, 0.1, 0.1], generator=_gen)
