import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, InterpolationMode


class TiledBBBC038Dataset(Dataset):
    """
    Wraps an existing BBBC038Dataset (or any dataset with the same
    __getitem__ contract) and splits each full-resolution sample into
    an N x N grid of overlapping tiles, each resized to img_size.

    Length goes from S -> S * N^2. Tiling requires access to the
    full-resolution image, so the wrapped dataset's samples are loaded
    then re-resized per tile (the base dataset's own img_size resize
    is not used).
    """

    def __init__(self, base_dataset, n_tiles: int = 2, overlap: float = 0.1):
        self.base_dataset = base_dataset
        self.n_tiles = n_tiles
        self.overlap = overlap
        self.img_size = base_dataset.img_size
        self.is_training = base_dataset.is_training

    def __len__(self):
        return len(self.base_dataset) * self.n_tiles ** 2

    def _tile_bounds(self, h: int, w: int):
        n = self.n_tiles
        stride_h, stride_w = h / n, w / n
        pad_h, pad_w = stride_h * self.overlap, stride_w * self.overlap

        bounds = []
        for i in range(n):
            y0 = max(0, int(i * stride_h - pad_h))
            y1 = min(h, int((i + 1) * stride_h + pad_h))
            for j in range(n):
                x0 = max(0, int(j * stride_w - pad_w))
                x1 = min(w, int((j + 1) * stride_w + pad_w))
                bounds.append((y0, y1, x0, x1))
        return bounds

    def _resize(self, tensor, mode):
        needs_batch = tensor.dim() == 2
        if needs_batch:
            tensor = tensor.unsqueeze(0)
        tensor = resize(tensor, [self.img_size, self.img_size], interpolation=mode,
                         antialias=(mode == InterpolationMode.BILINEAR))
        return tensor[0] if needs_batch else tensor

    def __getitem__(self, idx):
        n2 = self.n_tiles ** 2
        sample_idx, tile_idx = idx // n2, idx % n2

        sample = self.base_dataset[sample_idx]
        sample_id = sample[0]
        image = sample[1]

        _, h, w = image.shape
        y0, y1, x0, x1 = self._tile_bounds(h, w)[tile_idx]
        tile_id = f'{sample_id}_tile{tile_idx}'

        tile_image = self._resize(image[:, y0:y1, x0:x1], InterpolationMode.BILINEAR)

        if not self.is_training:
            return tile_id, tile_image

        semantic_mask, instance_map = sample[2], sample[3]

        tile_instance = self._resize(instance_map[y0:y1, x0:x1], InterpolationMode.NEAREST)
        tile_semantic = (tile_instance > 0).float()

        return tile_id, tile_image, tile_semantic, tile_instance
