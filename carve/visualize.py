from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .attention import dynamic_reshape_attention


def _heat(arr: np.ndarray, size) -> np.ndarray:
    a = arr - arr.min()
    a = a / (a.max() + 1e-8)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.BILINEAR))


def plot_pipeline(image, out, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(image)
    axes[1].imshow(_heat(out.sum_att_map, image.size), cmap="jet", alpha=0.5,
                   extent=(0, image.size[0], image.size[1], 0))
    axes[1].set_title("Contrastive attention")
    axes[2].imshow(out.mask_map, cmap="gray")
    axes[2].set_title("Top-K mask")
    axes[3].imshow(out.refined_image)
    axes[3].set_title("Refined image")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_layerwise(out, lam=1e-8, step=0, save_path=None):
    import matplotlib.pyplot as plt

    grids, layers = [], []
    q, g = out.query_attention[step], out.general_attention[step]
    for l in range(len(q)):
        grid, _ = dynamic_reshape_attention(q[l] / (g[l] + lam), out.output_shape)
        if grid is not None:
            grids.append(grid)
            layers.append(l)

    n = len(grids)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(grids[i], cmap="viridis", interpolation="nearest")
            ax.set_title(f"Layer {layers[i]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
