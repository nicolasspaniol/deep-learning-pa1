from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize, InterpolationMode


class BBBC038Dataset(Dataset):
    def __init__(self, is_training: bool, path: str | Path, img_size: int = 128):
        self.data_dir = Path(path)

        if not self.data_dir.exists():
            raise RuntimeError(f'Dataset not found at "{self.data_dir}"')

        self.img_size = img_size
        self.is_training = is_training

        # Each subfolder of data_dir is one sample ID. Ignore files such as
        # archive metadata that may be present alongside the samples.
        self.ids = sorted(path.name for path in self.data_dir.iterdir() if path.is_dir())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample_dir = self.data_dir / sample_id
        masks_dir = sample_dir / 'masks'
        img_path = sample_dir / 'images' / f'{sample_id}.png'

        # load image, drop alpha channel if present, normalize to [0, 1]
        image = read_image(str(img_path))[:3].float() / 255.0

        # resize image (interpolated)
        image = resize(
            image,
            [self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        # if loading the test dataset, returns only the input image
        if not self.is_training:
            return sample_id, image

        # rest of the method considers is_training=True

        mask_files = sorted(path for path in masks_dir.iterdir() if path.is_file())

        instance_map = torch.zeros(
            (self.img_size, self.img_size), dtype=torch.int64
        )

        # merges all instance masks into a single map, with each instance having
        # its own ID
        for instance_id, mask_path in enumerate(mask_files, 1):
            mask = read_image(str(mask_path))[0] > 0
            mask = resize(
                mask.unsqueeze(0),
                (self.img_size, self.img_size),
                interpolation=InterpolationMode.NEAREST,
            )[0]
            instance_map[mask] = instance_id

        semantic_mask = (instance_map > 0).float()

        return sample_id, image, semantic_mask, instance_map
