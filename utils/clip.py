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


# -----------------------------
# Result container
# -----------------------------

@dataclass
class CLIPResult:
    dataset: Path
    num_images: int
    global_mean: float
    per_model: dict[str, float]

    def to_json(self):
        return {
            "dataset": str(self.dataset),
            "num_images": self.num_images,
            "global_mean": self.global_mean,
            "per_model": self.per_model,
        }


# -----------------------------
# Text helpers
# -----------------------------

def chunk_prompt(tokenizer, text: str, max_len: int = 77, stride: int = 50) -> List[torch.Tensor]:
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


# -----------------------------
# Core
# -----------------------------

def run_clip(
    root: Path,
    csv_path: Path | None = None,
    output: Path | None = None,
    model_name: str = "openai/clip-vit-large-patch14",
    device: str | None = None,
    batch_size: int = 32,
) -> CLIPResult:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device == "cuda"

    if csv_path is None:
        csv_path = root / "SFHQ-T2I" / "SFHQ_T2I_dataset.csv"

    sorted_dir = root / "sorted"
    df = pd.read_csv(csv_path)

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    if use_fp16:
        model = model.half()

    tokenizer = processor.tokenizer

    scores = []
    per_model = {}

    rows = list(df.itertuples())

    for start in tqdm(range(0, len(rows), batch_size)):

        batch_rows = rows[start : start + batch_size]

        images = []
        text_chunks = []
        chunk_map = []
        model_names = []

        for idx, row in enumerate(batch_rows):
            img_path = sorted_dir / row.model_used / row.image_filename

            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            chunks = chunk_prompt(tokenizer, row.text_prompt)

            if not chunks:
                continue

            images.append(image)
            model_names.append(row.model_used)

            for c in chunks:
                text_chunks.append(c)
                chunk_map.append(len(images) - 1)

        if not images:
            continue

        image_inputs = processor(images=images, return_tensors="pt").to(device, non_blocking=True)

        text_inputs = tokenizer.pad(
            {"input_ids": text_chunks},
            return_tensors="pt",
        ).to(device, non_blocking=True)

        with torch.inference_mode():

            outputs = model(
                pixel_values=image_inputs["pixel_values"],
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )

            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            sims = txt_emb @ img_emb.T

        # aggregate sliding window → MAX
        for img_idx in range(len(images)):

            chunk_indices = [i for i, m in enumerate(chunk_map) if m == img_idx]

            if not chunk_indices:
                continue

            score = sims[chunk_indices, img_idx].max().item()

            scores.append(score)

            per_model.setdefault(model_names[img_idx], []).append(score)

    per_model_mean = {
        m: float(torch.tensor(v).mean()) for m, v in per_model.items()
    }

    result = CLIPResult(
        dataset=root,
        num_images=len(scores),
        global_mean=float(torch.tensor(scores).mean()),
        per_model=per_model_mean,
    )

    if output:
        output.write_text(json.dumps(result.to_json(), indent=2))

    return result
