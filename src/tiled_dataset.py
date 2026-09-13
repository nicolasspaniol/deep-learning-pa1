import torch
from torch.utils.data import Dataset


class TiledBBBC038Dataset(Dataset):
    """
    Wraps an existing BBBC038Dataset (or any dataset with the same
    __getitem__ contract) and splits each sample into a 4x4 grid of
    overlapping tiles.

    With img_size=128, n_tiles=4 (stride=32) and overlap=0.25 (pad=8 per
    side), every tile is exactly 48x48 -- a multiple of 8, as required by
    the ResUNet's stride-2 downsampling. Edge tiles are shifted inward
    (instead of clipped) so every tile keeps the same size with no
    resizing or padding needed anywhere.

    Length goes from S -> S * 16.
    """

    N_TILES = 4
    OVERLAP = 0.25

    def __init__(self, base_dataset):
        img_size = base_dataset.img_size
        assert img_size % self.N_TILES == 0, \
            f"img_size ({img_size}) must divide evenly by {self.N_TILES}"

        stride = img_size // self.N_TILES
        pad = int(stride * self.OVERLAP)
        tile_size = stride + 2 * pad
        assert tile_size % 8 == 0, \
            f"tile_size ({tile_size}) must be a multiple of 8 -- adjust N_TILES/OVERLAP"

        self.base_dataset = base_dataset
        self.n_tiles = self.N_TILES
        self.img_size = img_size
        self.is_training = base_dataset.is_training
        self.stride = stride
        self.tile_size = tile_size

    def __len__(self):
        return len(self.base_dataset) * self.n_tiles ** 2

    def _tile_bounds(self, tile_idx):
        i, j = divmod(tile_idx, self.n_tiles)

        def window(k):
            start = k * self.stride - (self.tile_size - self.stride) // 2
            start = max(0, min(start, self.img_size - self.tile_size))
            return start, start + self.tile_size

        y0, y1 = window(i)
        x0, x1 = window(j)
        return y0, y1, x0, x1

    def __getitem__(self, index):
        n2 = self.n_tiles ** 2
        sample_idx, tile_idx = index // n2, index % n2

        sample = self.base_dataset[sample_idx]
        sample_id = sample[0]
        image = sample[1]

        y0, y1, x0, x1 = self._tile_bounds(tile_idx)
        tile_id = f'{sample_id}_tile{tile_idx}'

        tile_image = image[:, y0:y1, x0:x1]

        if not self.is_training:
            return tile_id, tile_image

        instance_map = sample[3]

        tile_instance = instance_map[y0:y1, x0:x1]
        tile_semantic = (tile_instance > 0).float()

        return tile_id, tile_image, tile_semantic, tile_instance
