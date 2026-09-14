from src.bbbc038_dataset import BBBC038Dataset
from src.center_offset_dataset import CenterOffsetDataset
from src.tiled_dataset import TiledBBBC038Dataset

from torch import Generator
from torch.utils.data import random_split, Dataset


_gen = Generator().manual_seed(42)
_img_size = 128


# workaround to use random_split and still have tiles from the same image in sequence
class _TileIndexView(Dataset):
    def __init__(self, tiled_dataset, sample_indices):
        self.tiled = tiled_dataset
        self.sample_indices = list(sample_indices)
        self.n2 = tiled_dataset.n_tiles ** 2

    def __getattr__(self, name):
        return getattr(self.tiled, name)

    def __len__(self):
        return len(self.sample_indices) * self.n2

    def __getitem__(self, idx):
        sample_pos, tile_idx = divmod(idx, self.n2)
        sample_idx = self.sample_indices[sample_pos]
        return self.tiled[sample_idx * self.n2 + tile_idx]


def bbbc038_center_offsets():
    dataset = CenterOffsetDataset(BBBC038Dataset(True, './data/stage1_train', img_size=_img_size))
    return random_split(dataset, [0.8, 0.1, 0.1], generator=_gen)


# see _TileIndexView
def bbbc038_center_offsets_tiled():
    base = BBBC038Dataset(True, './data/stage1_train', img_size=_img_size)
    tiled = TiledBBBC038Dataset(base)

    n_samples = len(base)
    train_idx, val_idx, test_idx = random_split(
        range(n_samples), [0.8, 0.1, 0.1], generator=_gen
    )

    return (
        CenterOffsetDataset(_TileIndexView(tiled, train_idx)),
        CenterOffsetDataset(_TileIndexView(tiled, val_idx)),
        CenterOffsetDataset(_TileIndexView(tiled, test_idx)),
    )
