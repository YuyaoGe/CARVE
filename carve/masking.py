from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def create_attention_mask(
    sum_att_map: np.ndarray,
    *,
    top_percent: float = 0.2,
    min_keep_regions: int = 3,
    outlier_percent: float = 0.1,
    strategy: str = "attention_sum",
) -> np.ndarray:
    from scipy import ndimage

    threshold = np.percentile(sum_att_map, (1.0 - top_percent) * 100)
    mask_map = (sum_att_map >= threshold).astype(np.uint8)

    labeled, num = ndimage.label(mask_map)
    if num == 0:
        return np.zeros_like(sum_att_map, dtype=np.uint8)

    if strategy == "remove_small":
        sizes = ndimage.sum(mask_map, labeled, range(1, num + 1))
        total = int(mask_map.sum())
        min_size = int(total * outlier_percent / 100) if total > 0 else 1
        for i, size in enumerate(sizes, 1):
            if size < min_size:
                mask_map[labeled == i] = 0
        return mask_map

    if strategy == "size":
        scores = ndimage.sum(mask_map, labeled, range(1, num + 1))
    elif strategy == "attention_sum":
        scores = ndimage.sum(sum_att_map, labeled, range(1, num + 1))
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    ranked = sorted(((i + 1, s) for i, s in enumerate(scores)),
                    key=lambda x: x[1], reverse=True)
    keep_count = max(min_keep_regions, int(num * top_percent))
    keep = {rid for rid, _ in ranked[:keep_count]}

    new_mask = np.zeros_like(mask_map)
    for rid in keep:
        new_mask[labeled == rid] = 1
    return new_mask


def apply_mask_to_image(image: Image.Image, mask_map: np.ndarray) -> Image.Image:
    arr = np.array(image)
    masked = arr * np.stack([mask_map] * 3, axis=-1)
    return Image.fromarray(masked.astype(np.uint8))


def get_crop_bbox_from_mask(mask_map: np.ndarray):
    ys, xs = np.where(mask_map == 1)
    if len(ys) == 0:
        return 0, 0, mask_map.shape[1], mask_map.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_crop_to_match_original(crop_image: Image.Image, original_size):
    cw, ch = crop_image.size
    ow, oh = original_size
    scale = max(ow / cw, oh / ch)
    return crop_image.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)


def att_sum_crop_pipeline(
    sum_att_map: np.ndarray,
    image: Image.Image,
    *,
    top_percent: float = 0.2,
    min_keep_regions: int = 3,
    outlier_percent: float = 0.1,
    strategy: str = "attention_sum",
    save_dir: str = None,
):
    mask_map = create_attention_mask(
        sum_att_map, top_percent=top_percent, min_keep_regions=min_keep_regions,
        outlier_percent=outlier_percent, strategy=strategy,
    )
    masked_image = apply_mask_to_image(image, mask_map)
    bbox = get_crop_bbox_from_mask(mask_map)
    crop_masked_image = resize_crop_to_match_original(
        masked_image.crop(bbox), image.size
    )

    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)
        norm = (sum_att_map - sum_att_map.min()) / (sum_att_map.ptp() + 1e-8)
        Image.fromarray((norm * 255).astype(np.uint8)).save(
            os.path.join(save_dir, "sum_att_map.png"))
        Image.fromarray((mask_map * 255).astype(np.uint8)).save(
            os.path.join(save_dir, "mask_map.png"))
        masked_image.save(os.path.join(save_dir, "masked_image.png"))
        crop_masked_image.save(os.path.join(save_dir, "crop_masked_image.png"))
        marked = image.copy()
        ImageDraw.Draw(marked).rectangle(bbox, outline="red", width=5)
        marked.save(os.path.join(save_dir, "marked_original.png"))

    return {
        "crop_masked_image": crop_masked_image,
        "sum_att_map": sum_att_map,
        "mask_map": mask_map,
        "masked_image": masked_image,
        "bbox": bbox,
    }
