import torch
from torch.utils.data import Dataset


class CenterOffsetDataset(Dataset):
    def __init__(self, base_dataset, gaussian_sigma: float = 2.0):
        assert base_dataset.is_training, "CenterOffsetDataset needs the training split (needs instance_map)"
        self.base_dataset = base_dataset
        self.gaussian_sigma = gaussian_sigma
        # no more precomputed _yy/_xx here

    def __len__(self):
        return len(self.base_dataset)

    def _build_center_targets(self, instance_map: torch.Tensor, sigma: float):
        H, W = instance_map.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij',
        )

        heatmap = torch.zeros((H, W), dtype=torch.float32)
        offsets = torch.zeros((2, H, W), dtype=torch.float32)
        offset_mask = torch.zeros((H, W), dtype=torch.float32)

        instance_ids = torch.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        inv_two_sigma_sq = 1.0 / (2 * sigma ** 2)

        for inst_id in instance_ids.tolist():
            inst_mask = instance_map == inst_id
            ys, xs = torch.nonzero(inst_mask, as_tuple=True)
            if ys.numel() == 0:
                continue

            cy = ys.float().mean()
            cx = xs.float().mean()

            dxx = xx - cx
            dyy = yy - cy
            dist_sq = dxx ** 2 + dyy ** 2
            blob = torch.exp(-dist_sq * inv_two_sigma_sq)
            heatmap = torch.maximum(heatmap, blob)

            offsets[0][inst_mask] = cx - xx[inst_mask]
            offsets[1][inst_mask] = cy - yy[inst_mask]
            offset_mask[inst_mask] = 1.0

        return heatmap.unsqueeze(0), offsets, offset_mask.unsqueeze(0)

    def __getitem__(self, index):
        sample_id, image, semantic_mask, instance_map = self.base_dataset[index]
        heatmap, offsets, offset_mask = self._build_center_targets(instance_map, self.gaussian_sigma)
        return sample_id, image, heatmap, offsets, offset_mask
