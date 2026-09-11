from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import torch


class SyntheticEllipseDataset(Dataset):
    def __init__(self, num_samples=500, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def _generate_sample(self):
        size = self.img_size
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        # OpenCV drawing functions support int32 labels, which are converted
        # to the int64 tensor type used by BBBC038Dataset before returning.
        instance_map = np.zeros((size, size), dtype=np.int32)

        num_ellipses = random.randint(5, 20)
        for instance_id in range(1, num_ellipses + 1):
            center = (random.randint(10, size-10), random.randint(10, size-10))
            axes = (random.randint(5, 15), random.randint(5, 15))
            angle = random.randint(0, 180)
            gray = random.randint(10, 180)

            cv2.ellipse(img, center, axes, angle, 0, 360, (gray, gray, gray), -1)
            # Keep the individual ellipse ID, matching the instance-map
            # convention used by BBBC038Dataset. Later ellipses own pixels in
            # overlap regions, just as later mask files do in that dataset.
            cv2.ellipse(instance_map, center, axes, angle, 0, 360, instance_id, -1)

        noise = np.random.normal(0, 30, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img, instance_map

    def __getitem__(self, idx):
        img, instance_map = self._generate_sample()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        instance_map_t = torch.from_numpy(instance_map).to(torch.int64)
        semantic_mask_t = (instance_map_t > 0).float()

        return img_t, semantic_mask_t, instance_map_t
