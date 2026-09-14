import torch
import torchvision.transforms.functional as TF


class CorruptedBBBC038Dataset:
    """
    Wraps a BBBC038Dataset (or TiledBBBC038Dataset, same __getitem__ contract)
    and applies an image-only corruption at a fixed intensity level.
    Masks / instance maps are passed through unchanged, so it can be dropped
    straight into evaluate_instance_prediction / compute_map without changes.

    corruption: one of 'blur', 'noise', 'brightness_contrast'
    intensity:  1, 2 or 3 (mild -> strong), used to sweep a degradation curve
    """

    _BLUR_SIGMA = {1: 1.0, 2: 2.0, 3: 4.0}
    _NOISE_STD = {1: 0.05, 2: 0.10, 3: 0.20}
    # (brightness_factor, contrast_factor) -- <1 darkens/flattens, matches
    # the "worse at higher intensity" convention used above
    _BRIGHTNESS_CONTRAST = {
        1: (0.85, 0.85),
        2: (0.65, 0.65),
        3: (0.45, 0.45),
    }

    def __init__(self, base_dataset, corruption: str, intensity: int):
        assert corruption in ('blur', 'noise', 'brightness_contrast'), \
            "corruption must be 'blur', 'noise' or 'brightness_contrast'"
        assert intensity in (1, 2, 3), "intensity must be 1, 2 or 3"

        self.base_dataset = base_dataset
        self.corruption = corruption
        self.intensity = intensity

    def __len__(self):
        return len(self.base_dataset)

    def _apply_blur(self, image):
        sigma = self._BLUR_SIGMA[self.intensity]
        kernel_size = int(2 * round(3 * sigma) + 1)  # odd, ~3 sigma radius
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    def _apply_noise(self, image):
        std = self._NOISE_STD[self.intensity]
        noise = torch.randn_like(image) * std
        return (image + noise).clamp(0.0, 1.0)

    def _apply_brightness_contrast(self, image):
        brightness, contrast = self._BRIGHTNESS_CONTRAST[self.intensity]
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        return image.clamp(0.0, 1.0)

    def _corrupt(self, image):
        if self.corruption == 'blur':
            return self._apply_blur(image)
        if self.corruption == 'noise':
            return self._apply_noise(image)
        return self._apply_brightness_contrast(image)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        sample_id, image = sample[0], sample[1]
        corrupted_image = self._corrupt(image)

        return (sample_id, corrupted_image) + tuple(sample[2:])

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)
