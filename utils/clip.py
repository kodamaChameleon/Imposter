"""
-*- coding: utf-8 -*-
Name:        utils/config.py
Purpose:     CLIP score computation for SFHQ-T2I.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""

import csv
from dataclasses import dataclass, fields, asdict
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
    dataset: str
    clip_mode: str
    num_images: int
    score: float


# -----------------------------
# Helpers
# -----------------------------


def append_clip_csv(results: List[CLIPResult], csv_path: Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    fieldnames = [f.name for f in fields(CLIPResult)]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for r in results:
            writer.writerow(asdict(r))


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


def aggregate_sliding(sims, chunk_map, model_names, per_model):
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

        per_model.setdefault(model_names[img_idx], []).append(score)


def aggregate_truncate(sims, model_names, per_model):
    """
    Aggregate similarities when using truncated prompts.
    Assumes one text per image and uses the diagonal of the similarity matrix.
    """
    for i, score in enumerate(sims.diag()):
        val = score.item()
        per_model.setdefault(model_names[i], []).append(val)


# -----------------------------
# Core
# -----------------------------


def run_clip(
    root: Path,
    dataset_csv: Path | None = None,   # ← was csv_path (INPUT)
    results_csv: Path | None = None,   # ← NEW (OUTPUT, append)
    model_name: str = CLIP_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    clip_mode: str = "sliding",
) -> List[CLIPResult]:
    """
    Compute CLIP image–text similarity scores for a generated image dataset.

    Parameters
    ----------
    root : Path | `root / sorted / <model_name> / <image_file>`.
    dataset_csv : Path, optional
    results_csv : Path, optional | If provided, writes the aggregated result as csv.
    model_name : str | Hugging Face CLIP model identifier.
    device : str, optional | Target device. Auto-selects CUDA if available.
    batch_size : int, default=32 | Number of samples processed per forward pass.
    clip_mode : {"sliding", "truncate"}
    """

    if clip_mode not in {"sliding", "truncate"}:
        raise ValueError(f"Invalid clip_mode: {clip_mode}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device == "cuda"

    if dataset_csv is None:
        dataset_csv = root / "SFHQ-T2I" / "SFHQ_T2I_dataset.csv"

    df = pd.read_csv(dataset_csv)
    sorted_dir = root / "sorted"

    model, processor, tokenizer = load_clip(model_name, device, use_fp16)

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
            except Exception as e:
                print(f"WARNING: {e}")
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
            aggregate_sliding(sims, chunk_map, model_names, per_model)
        elif clip_mode == "truncate":
            aggregate_truncate(sims, model_names, per_model)

    results: List[CLIPResult] = []

    for dataset_name, values in per_model.items():
        results.append(
            CLIPResult(
                dataset=dataset_name,
                clip_mode=clip_mode,
                num_images=len(values),
                score = float(sum(values) / len(values)),
            )
        )

    if results_csv is not None:
        append_clip_csv(results, results_csv)

    return results
