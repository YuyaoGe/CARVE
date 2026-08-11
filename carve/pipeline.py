from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from PIL import Image

from .attention import (
    DEFAULT_LAYERS,
    GENERAL_INSTRUCTION,
    fuse_contrastive_attention,
    generate_with_attention,
)
from .masking import att_sum_crop_pipeline

ANSWER_SUFFIX = "Answer the question using a single word or phrase."

VERSIONS = {
    "v1": ("first", "remove_small"),
    "v2": ("first", "size"),
    "v3": ("first", "attention_sum"),
    "v4": ("all", "attention_sum"),
}


@dataclass
class CarveOutput:
    original_answer: str
    refined_answer: str
    sum_att_map: "object"
    mask_map: "object"
    refined_image: Image.Image
    bbox: tuple
    query_attention: list
    general_attention: list
    output_shape: tuple


def load_model(model_id: str, family: str = "qwen2_5", device: str = "cuda"):
    from transformers import AutoProcessor

    if family == "qwen2_5":
        from transformers import Qwen2_5_VLForConditionalGeneration

        max_pixels = 256 * 28 * 28
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels, use_fast=True)
        processor.image_processor.size["longest_edge"] = max_pixels
    elif family == "llava":
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    else:
        raise ValueError(f"Unsupported family {family!r}")
    return model, processor


def _vision_token_ids(processor, family):
    if family == "qwen2_5":
        tok = processor.tokenizer
        return (tok.convert_tokens_to_ids("<|vision_start|>"),
                tok.convert_tokens_to_ids("<|vision_end|>"))
    return (32000, None)


def _build_inputs(processor, family, images, text, device):
    query = f"{text} {ANSWER_SUFFIX}"
    if family == "qwen2_5":
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[chat], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
    else:
        tag = "<image>" * len(images)
        prompt = f"{tag}\nUSER: {query}\nASSISTANT:"
        inputs = processor(images, prompt, return_tensors="pt", padding=True)
    return inputs.to(device)


def _output_shape(processor, family, image):
    if family == "qwen2_5":
        aux = processor.image_processor(images=[image])
        if "image_grid_thw" in aux:
            hw = aux["image_grid_thw"].numpy().squeeze(0)[1:] / 2
            return tuple(hw.astype(int))
    return (24, 24)


def carve(
    model,
    processor,
    image: Image.Image,
    question: str,
    *,
    family: str = "qwen2_5",
    version: str = "v4",
    layers: Sequence[int] = DEFAULT_LAYERS,
    lam: float = 1e-8,
    top_percent: float = 0.2,
    min_keep_regions: int = 3,
    outlier_percent: float = 0.1,
    max_new_tokens: int = 32,
    general_instruction: str = GENERAL_INSTRUCTION,
    device: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> CarveOutput:
    if version not in VERSIONS:
        raise ValueError(f"Unknown version {version!r}; expected {list(VERSIONS)}")
    step_mode, strategy = VERSIONS[version]

    image = image.convert("RGB")
    device = device or model.device
    start_id, end_id = _vision_token_ids(processor, family)
    out_shape = _output_shape(processor, family, image)

    q_inputs = _build_inputs(processor, family, [image], question, device)
    g_inputs = _build_inputs(processor, family, [image], general_instruction, device)

    q_res = generate_with_attention(
        model, processor, q_inputs, max_new_tokens=max_new_tokens,
        vision_start_token_id=start_id, vision_end_token_id=end_id,
    )
    g_res = generate_with_attention(
        model, processor, g_inputs, max_new_tokens=max_new_tokens,
        vision_start_token_id=start_id, vision_end_token_id=end_id,
    )

    sum_att_map = fuse_contrastive_attention(
        q_res["all_attention_maps"], g_res["all_attention_maps"], image.size,
        layers=layers, lam=lam, output_shape=out_shape, step_mode=step_mode,
    )

    result = att_sum_crop_pipeline(
        sum_att_map, image, top_percent=top_percent,
        min_keep_regions=min_keep_regions, outlier_percent=outlier_percent,
        strategy=strategy, save_dir=save_dir,
    )
    refined = result["crop_masked_image"]

    combined = _build_inputs(processor, family, [image, refined], question, device)
    with torch.no_grad():
        gen_ids = model.generate(**combined, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = gen_ids[:, combined["input_ids"].shape[1]:]
    refined_answer = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    if "ASSISTANT:" in refined_answer:
        refined_answer = refined_answer.split("ASSISTANT:")[-1].strip()

    return CarveOutput(
        original_answer=q_res["generated_text"],
        refined_answer=refined_answer,
        sum_att_map=sum_att_map,
        mask_map=result["mask_map"],
        refined_image=refined,
        bbox=result["bbox"],
        query_attention=q_res["all_attention_maps"],
        general_attention=g_res["all_attention_maps"],
        output_shape=out_shape,
    )
