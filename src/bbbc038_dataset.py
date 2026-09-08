import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.transform import resize

from utils import is_binary

BBBC038_URL = 'https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip'


class BBBC038Dataset(Dataset):
    def __init__(self, root_dir='./data', img_size=128, download=False):
        '''
        Args:
            root_dir: base folder where the dataset lives (or will be downloaded to).
            img_size: images/masks are resized to (img_size, img_size).
            download: if True, download + extract the dataset when it's not
                      already present at `root_dir/stage1_train`.
        '''
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / 'stage1_train'
        self.img_size = img_size

        if download:
            self._download()

        if not self.train_dir.exists():
            raise RuntimeError(
                f'Dataset not found at {self.train_dir}. '
                'Pass download=True to fetch it automatically.'
            )

        # each subfolder of train_dir is one sample id
        self.ids = sorted(os.listdir(self.train_dir))

    def _download(self):
        '''Download and extract the BBBC038 stage1_train set if not already present.'''
        if self.train_dir.exists():
            return  # already downloaded, nothing to do

        self.root_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_dir / 'stage1_train.zip'

        if not zip_path.exists():
            print(f'Downloading BBBC038 dataset to {zip_path} ...')
            urllib.request.urlretrieve(BBBC038_URL, zip_path)

        print(f'Extracting {zip_path} ...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.train_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample_dir = self.train_dir / sample_id
        mask_dir = sample_dir / 'masks'
        img_path = sample_dir / 'images' / f'{sample_id}.png'

        # load image, drop alpha channel if present, normalize to [0, 1]
        image = imread(img_path)[..., :3] / 255.0

        # each sample has multiple instance masks; merge them into one binary mask
        masks = [imread(mask_dir / f) for f in os.listdir(mask_dir)]
        mask = np.max(masks, axis=0) / 255

        # resize image (interpolated) and mask (nearest-neighbor, no anti-aliasing)
        image = resize(image, (self.img_size, self.img_size), preserve_range=True)
        mask = resize(
            mask,
            (self.img_size, self.img_size),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

        # to CHW tensors
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        mask = torch.from_numpy(mask[..., None]).permute(2, 0, 1)[0].contiguous().float()

        return image, mask


class InstanceEvaluationSubset(Dataset):
    def __init__(self, original_dataset, indices, img_size=128):
        self.original_dataset = original_dataset
        self.indices = list(indices)
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]

        image, semantic_mask = self.original_dataset[original_index]

        image_id = self.original_dataset.ids[original_index]
        mask_dir = (
            self.original_dataset.train_dir
            / image_id
            / "masks"
        )

        instance_map = np.zeros(
            (self.img_size, self.img_size)
        )

        mask_files = sorted(os.listdir(mask_dir))

        for instance_id, mask_file in enumerate( mask_files, start=1):
            instance_mask = imread( mask_dir/mask_file) > 0

            instance_mask = resize( instance_mask.astype(np.uint8), (self.img_size, self.img_size),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(bool)

            instance_map[instance_mask] = instance_id

        instance_map = torch.from_numpy(
            instance_map
        ).long()

        return (
            image,
            semantic_mask,
            instance_map,
            image_id
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = BBBC038Dataset(download=True)
    print(f'Loaded {len(ds)} samples')

    for img, mask in DataLoader(ds, shuffle=True):
        img, mask = img[0], mask[0]

        assert img.shape == (3, ds.img_size, ds.img_size), img.shape
        assert mask.shape == (ds.img_size, ds.img_size), mask.shape
        assert img.dtype == torch.float32, img.dtype
        assert mask.dtype == torch.float32, mask.dtype
        assert img.min() >= 0 and img.max() <= 1, (img.min(), img.max())
        assert is_binary(mask)

        img = img.permute(1, 2, 0).cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Answer (0/1 mask)")
        axes[1].axis("off")

        plt.show()
