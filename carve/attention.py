from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

GENERAL_INSTRUCTION = "Write a general description of the image."

DEFAULT_LAYERS = list(range(20, 26))


def _resize(arr: np.ndarray, size) -> np.ndarray:
    return np.asarray(
        Image.fromarray(arr.astype(np.float32), mode="F").resize(size, Image.BILINEAR),
        dtype=np.float32,
    )


def generate_with_attention(
    model,
    processor,
    inputs,
    *,
    max_new_tokens: int = 32,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    num_image_tokens: int = 576,
):
    import torch

    ids = inputs["input_ids"].tolist()[0]
    if vision_start_token_id is not None and vision_start_token_id in ids:
        if vision_end_token_id is not None:
            pos = ids.index(vision_start_token_id) + 1
            pos_end = ids.index(vision_end_token_id)
        else:
            pos = ids.index(vision_start_token_id)
            pos_end = pos + num_image_tokens
    else:
        pos, pos_end = 1, 2

    eos_token_id = processor.tokenizer.eos_token_id
    input_length = inputs["input_ids"].shape[1]
    cur_ids = inputs["input_ids"].clone()
    cur_mask = inputs["attention_mask"].clone()
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}

    all_attention_maps: List[List[np.ndarray]] = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=cur_ids, attention_mask=cur_mask,
                output_attentions=True, **extra,
            )
        next_token = torch.argmax(outputs.logits[0, -1, :]).view(1, 1)

        step_maps = []
        for layer_attention in outputs.attentions:
            att = layer_attention[0, :, -1, pos:pos_end].mean(dim=0)
            step_maps.append(att.to(torch.float32).detach().cpu().numpy())
        all_attention_maps.append(step_maps)

        if next_token.item() == eos_token_id:
            break
        cur_ids = torch.cat([cur_ids, next_token], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones((1, 1), device=cur_mask.device)], dim=1
        )

    generated = processor.tokenizer.decode(
        cur_ids[0][input_length:], skip_special_tokens=True
    ).strip()
    return {
        "generated_text": generated,
        "all_attention_maps": all_attention_maps,
        "vision_token_positions": (pos, pos_end),
    }


def dynamic_reshape_attention(att: np.ndarray, fallback_shape=None):
    size = att.shape[0] if att.ndim == 1 else att.size
    if size <= 1:
        return None, None

    known = {576: (24, 24), 1024: (32, 32), 1600: (40, 40), 1681: (41, 41), 256: (16, 16)}
    if size in known:
        return att.reshape(known[size]), known[size]

    if fallback_shape is not None:
        expected = fallback_shape[0] * fallback_shape[1]
        if size == expected:
            return att.reshape(fallback_shape), tuple(fallback_shape)
        if size > expected:
            return att[:expected].reshape(fallback_shape), tuple(fallback_shape)

    root = int(math.sqrt(size))
    if root * root == size and root > 1:
        return att.reshape(root, root), (root, root)
    return None, None


def fuse_contrastive_attention(
    query_maps: List[List[np.ndarray]],
    general_maps: List[List[np.ndarray]],
    image_size,
    *,
    layers: Sequence[int] = DEFAULT_LAYERS,
    lam: float = 1e-8,
    output_shape=None,
    step_mode: str = "all",
):
    w, h = image_size
    S = np.zeros((h, w), dtype=np.float64)
    steps = [0] if step_mode == "first" else range(min(len(query_maps), len(general_maps)))

    for step in steps:
        wt = (step + 1) if step_mode == "all" else 1.0
        layer_sum = np.zeros((h, w), dtype=np.float64)
        for l in layers:
            if l >= len(query_maps[step]) or l >= len(general_maps[step]):
                continue
            contrast = query_maps[step][l] / (general_maps[step][l] + lam)
            grid, _ = dynamic_reshape_attention(contrast, output_shape)
            if grid is not None:
                layer_sum += _resize(grid, (w, h))
        S += wt * layer_sum
    return S
