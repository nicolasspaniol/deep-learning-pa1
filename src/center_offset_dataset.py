import torch
from torch.utils.data import Dataset


class CenterOffsetDataset(Dataset):
    """
    Wraps/extends a BBBC038Dataset (Trilha C) to additionally produce:
      - heatmap:      (1, H, W) gaussian blob peaked at each instance center
      - offsets:      (2, H, W) dx, dy from each foreground pixel to its
                       instance's center
      - offset_mask:  (1, H, W) 1 on foreground pixels, 0 elsewhere (mask the
                       offset loss with this)
    """

    def __init__(self, base_dataset, gaussian_sigma: float = 4.0):
        assert base_dataset.is_training, "CenterOffsetDataset needs the training split (needs instance_map)"
        self.base_dataset = base_dataset
        self.gaussian_sigma = gaussian_sigma

        img_size = base_dataset.img_size
        yy, xx = torch.meshgrid(
            torch.arange(img_size, dtype=torch.float32),
            torch.arange(img_size, dtype=torch.float32),
            indexing='ij',
        )
        self._yy = yy
        self._xx = xx

    def __len__(self):
        return len(self.base_dataset)

    def _build_center_targets(self, instance_map: torch.Tensor, alpha: float = 2.0, eps: float = 1e-2):
        H, W = instance_map.shape
        heatmap = torch.zeros((H, W), dtype=torch.float32)
        offsets = torch.zeros((2, H, W), dtype=torch.float32)
        offset_mask = torch.zeros((H, W), dtype=torch.float32)

        instance_ids = torch.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids.tolist():
            inst_mask = instance_map == inst_id
            ys, xs = torch.nonzero(inst_mask, as_tuple=True)
            if ys.numel() == 0:
                continue

            cy = ys.float().mean()
            cx = xs.float().mean()

            # matriz de covariância dos pixels da instância (encaixa a gaussiana na forma/orientação do objeto)
            dx = xs.float() - cx
            dy = ys.float() - cy
            n = xs.numel()

            var_x = (dx * dx).sum() / n + eps
            var_y = (dy * dy).sum() / n + eps
            cov_xy = (dx * dy).sum() / n

            cov = torch.tensor([[var_x, cov_xy], [cov_xy, var_y]])
            cov_scaled = alpha * cov  # escala pra controlar o "espalhamento" da gaussiana
            cov_inv = torch.linalg.inv(cov_scaled)

            # distância de Mahalanobis de cada pixel da imagem ao centro, usando a covariância da instância
            dxx = self._xx - cx
            dyy = self._yy - cy
            mdist_sq = (
                cov_inv[0, 0] * dxx ** 2
                + 2 * cov_inv[0, 1] * dxx * dyy
                + cov_inv[1, 1] * dyy ** 2
            )
            blob = torch.exp(-0.5 * mdist_sq)
            heatmap = torch.maximum(heatmap, blob)

            offsets[0][inst_mask] = cx - self._xx[inst_mask]
            offsets[1][inst_mask] = cy - self._yy[inst_mask]
            offset_mask[inst_mask] = 1.0

        return heatmap.unsqueeze(0), offsets, offset_mask.unsqueeze(0)

    def __getitem__(self, idx):
        sample_id, image, semantic_mask, instance_map = self.base_dataset[idx]
        heatmap, offsets, offset_mask = self._build_center_targets(instance_map)

        return sample_id, image, heatmap, offsets, offset_mask
