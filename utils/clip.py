"""
-*- coding: utf-8 -*-
Name:        utils/config.py
Purpose:     CLIP score computation for SFHQ-T2I.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL

# -----------------------------
# Result container
# -----------------------------


@dataclass
class CLIPResult:
    """
    Container for aggregated CLIP scoring results.
    """

    dataset: Path
    num_images: int
    global_mean: float
    per_model: dict[str, float]

    def to_json(self):
        """Serialize the result to a JSON-compatible dictionary."""
        return {
            "dataset": str(self.dataset),
            "num_images": self.num_images,
            "global_mean": self.global_mean,
            "per_model": self.per_model,
        }


# -----------------------------
# Helpers
# -----------------------------


def chunk_prompt(
    tokenizer, text: str, max_len: int = 77, stride: int = 50
) -> List[torch.Tensor]:
    """
    Tokenize a prompt into overlapping windows compatible with CLIP's
    maximum context length.
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"][0]

    chunks = []

    for i in range(0, len(tokens), stride):
        window = tokens[i : i + max_len - 2]

        if len(window) == 0:
            break

        window = torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id]),
                window,
                torch.tensor([tokenizer.eos_token_id]),
            ]
        )

        chunks.append(window)

        if i + max_len >= len(tokens):
            break

    return chunks


def load_clip(model_name: str, device: str, use_fp16: bool):
    """
    Load a pretrained CLIP model and its processor.
    """
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    if use_fp16:
        model = model.half()

    return model, processor, processor.tokenizer


def build_text_inputs_sliding(tokenizer, prompts, device):
    """
    Construct padded token batches for long prompts
    using sliding-window chunking.
    """
    text_chunks = []
    chunk_map = []

    for img_idx, text in enumerate(prompts):
        chunks = chunk_prompt(tokenizer, text)

        for c in chunks:
            text_chunks.append(c)
            chunk_map.append(img_idx)

    text_inputs = tokenizer.pad(
        {"input_ids": text_chunks},
        return_tensors="pt",
    ).to(device, non_blocking=True)

    return text_inputs, chunk_map


def build_text_inputs_truncate(processor, prompts, device):
    """
    Tokenize prompts using standard CLIP truncation.
    """
    return processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device, non_blocking=True)


def encode_batch(model, image_inputs, text_inputs):
    """
    Compute normalized CLIP embeddings and their cosine similarity matrix
    """
    with torch.inference_mode():
        outputs = model(
            pixel_values=image_inputs["pixel_values"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )

        img_emb = outputs.image_embeds.float()
        txt_emb = outputs.text_embeds.float()

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        sims = txt_emb @ img_emb.T

    return sims


def aggregate_sliding(sims, chunk_map, model_names, scores, per_model):
    """
    Aggregate sliding-window similarities into a single score per image.
    For each image, the maximum similarity across all its prompt chunks
    is selected.
    """
    for img_idx in range(len(model_names)):
        idxs = [i for i, m in enumerate(chunk_map) if m == img_idx]

        if not idxs:
            continue

        score = sims[idxs, img_idx].max().item()

        scores.append(score)
        per_model.setdefault(model_names[img_idx], []).append(score)


def aggregate_truncate(sims, model_names, scores, per_model):
    """
    Aggregate similarities when using truncated prompts.
    Assumes one text per image and uses the diagonal of the similarity matrix.
    """
    for i, score in enumerate(sims.diag()):
        val = score.item()
        scores.append(val)
        per_model.setdefault(model_names[i], []).append(val)


# -----------------------------
# Core
# -----------------------------


def run_clip(
    root: Path,
    csv_path: Path | None = None,
    output: Path | None = None,
    model_name: str = CLIP_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    clip_mode: str = "sliding",
) -> CLIPResult:
    """
    Compute CLIP image–text similarity scores for a generated image dataset.

    Parameters
    ----------
    root : Path | `root / sorted / <model_name> / <image_file>`.
    csv_path : Path, optional
    output : Path, optional | If provided, writes the aggregated result as JSON.
    model_name : str | Hugging Face CLIP model identifier.
    device : str, optional | Target device. Auto-selects CUDA if available.
    batch_size : int, default=32 | Number of samples processed per forward pass.
    clip_mode : {"sliding", "truncate"}
    """

    if clip_mode not in {"sliding", "truncate"}:
        raise ValueError(f"Invalid clip_mode: {clip_mode}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device == "cuda"

    if csv_path is None:
        csv_path = root / "SFHQ-T2I" / "SFHQ_T2I_dataset.csv"

    df = pd.read_csv(csv_path)
    sorted_dir = root / "sorted"

    model, processor, tokenizer = load_clip(model_name, device, use_fp16)

    scores = []
    per_model = {}

    rows = list(df.itertuples())

    for start in tqdm(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]

        images, prompts, model_names = [], [], []

        for row in batch:
            img_path = sorted_dir / row.model_used / row.image_filename

            if not img_path.exists():
                continue

            try:
                images.append(Image.open(img_path).convert("RGB"))
                prompts.append(row.text_prompt)
                model_names.append(row.model_used)
            except Exception:
                continue

        if not images:
            continue

        image_inputs = processor(images=images, return_tensors="pt").to(
            device, non_blocking=True
        )

        if clip_mode == "sliding":
            text_inputs, chunk_map = build_text_inputs_sliding(
                tokenizer, prompts, device
            )
        elif clip_mode == "truncate":
            text_inputs = build_text_inputs_truncate(processor, prompts, device)

        sims = encode_batch(model, image_inputs, text_inputs)

        if clip_mode == "sliding":
            aggregate_sliding(sims, chunk_map, model_names, scores, per_model)
        elif clip_mode == "truncate":
            aggregate_truncate(sims, model_names, scores, per_model)

    per_model_mean = {m: float(torch.tensor(v).mean()) for m, v in per_model.items()}

    result = CLIPResult(
        dataset=root,
        num_images=len(scores),
        global_mean=float(torch.tensor(scores).mean()),
        per_model=per_model_mean,
    )

    if output:
        output.write_text(json.dumps(result.to_json(), indent=2))

    return result
