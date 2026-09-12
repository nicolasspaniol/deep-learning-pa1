"""
Plotting utilities for heatmap + offset-field visualizations.

Design notes:
- Functions here take data in (arrays, tensors already moved to CPU) and an
  optional `ax`/`axes` to draw into. None of them call plt.show() or
  plt.savefig() -- the caller decides what to do with the figure.
- Pure computation (offsets_to_rgb, peak detection) lives outside the
  plotting functions so it can be tested/reused without matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Pure computation (no plotting, no torch dependency here)
# ---------------------------------------------------------------------------

def offsets_to_rgb(dx, dy, mask):
    """Encode a 2D offset field as an RGB image (hue=direction, value=magnitude)."""
    angle = np.arctan2(dy, dx)
    hue = (angle + np.pi) / (2 * np.pi)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag_max = mag.max() if mag.max() > 0 else 1.0
    value = np.clip(mag / mag_max, 0, 1)
    hsv = np.stack([hue, np.ones_like(hue), value], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[mask == 0] = 0.5
    return rgb


# ---------------------------------------------------------------------------
# Plotting building blocks
# ---------------------------------------------------------------------------

def plot_image_heatmap_offsets(axes_row, image, heatmap, dx, dy, mask,
                                title_prefix="", peaks=None):
    """
    Draw one row of 3 panels (image / heatmap overlay / offset field) into
    the given array of 3 Axes. Reusable for ground truth or predictions.

    axes_row: array-like of 3 matplotlib Axes
    peaks: optional (xs, ys) to scatter on top of the heatmap panel
    """
    ax_img, ax_heat, ax_off = axes_row

    ax_img.imshow(image)
    ax_img.set_title(f"{title_prefix}Image".strip())
    ax_img.axis('off')

    ax_heat.imshow(image)
    ax_heat.imshow(heatmap, cmap='hot', alpha=0.6)
    ax_heat.set_title(f"{title_prefix}Center heatmap".strip())
    ax_heat.axis('off')
    if peaks is not None:
        xs, ys = peaks
        ax_heat.scatter(xs, ys, s=30, c='k', marker='x')

    ax_off.imshow(offsets_to_rgb(dx, dy, mask))
    ax_off.set_title(f"{title_prefix}Offset field (hue=dir, brightness=mag)".strip())
    ax_off.axis('off')

    return axes_row


def plot_sample(sample, axes=None):
    """
    Plot a single dataset sample (image, GT heatmap, GT offset field).

    axes: optional array of 3 Axes to draw into. If None, creates its own
    figure. Returns the axes (does NOT call plt.show()).
    """
    _, image, heatmap, offsets, offset_mask = sample
    img_np = image.permute(1, 2, 0).numpy()
    heatmap_np = heatmap[0].numpy()
    dx, dy = offsets[0].numpy(), offsets[1].numpy()
    mask_np = offset_mask[0].numpy()

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))

    plot_image_heatmap_offsets(axes, img_np, heatmap_np, dx, dy, mask_np)
    plt.tight_layout()
    return axes


def plot_prediction(sample, pred_heatmap, pred_dx, pred_dy, pred_mask,
                     peaks=None, axes=None):
    """
    Plot GT (top row) vs model prediction (bottom row) for a sample.

    This takes already-computed prediction arrays rather than the model
    itself, so plotting stays independent of inference/model code.
    Use `run_inference` (in your model/inference module) to produce them.

    peaks: optional (xs, ys) of detected peak coordinates to overlay on the
    predicted heatmap panel.
    axes: optional (2, 3) array of Axes. If None, creates its own figure.
    """
    _, image, heatmap, offsets, offset_mask = sample
    img_np = image.permute(1, 2, 0).numpy()
    gt_heatmap_np = heatmap[0].numpy()
    gt_dx, gt_dy = offsets[0].numpy(), offsets[1].numpy()
    gt_mask_np = offset_mask[0].numpy()

    if axes is None:
        _, axes = plt.subplots(2, 3, figsize=(12, 8))

    # top row: ground truth (reuses the same building block as plot_sample)
    plot_image_heatmap_offsets(axes[0], img_np, gt_heatmap_np, gt_dx, gt_dy,
                                gt_mask_np, title_prefix="GT ")

    # bottom row: prediction
    plot_image_heatmap_offsets(axes[1], img_np, pred_heatmap, pred_dx, pred_dy,
                                pred_mask, title_prefix="Predicted ", peaks=peaks)

    plt.tight_layout()
    return axes
