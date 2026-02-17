"""
-*- coding: utf-8 -*-
Name:        utils/config.py
Purpose:     CLIP score computation for SFHQ-T2I.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import json
from dataclasses import dataclass
from pathlib import Path

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
# Core
# -----------------------------

def run_clip(
    root: Path,
    csv_path: Path | None = None,
    output: Path | None = None,
    model_name: str = "openai/clip-vit-large-patch14",
    device: str | None = None,
) -> CLIPResult:
    """
    Compute CLIP score for SFHQ-T2I sorted dataset.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if csv_path is None:
        csv_path = root / "SFHQ-T2I" / "SFHQ_T2I_dataset.csv"

    sorted_dir = root / "sorted"

    df = pd.read_csv(csv_path)

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    scores = []
    per_model = {}

    for row in tqdm(df.itertuples(), total=len(df)):
        img_path = sorted_dir / row.model_used / row.image_filename

        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            inputs = processor(
                text=[row.text_prompt],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

                score = (img_emb * txt_emb).sum().item()

            scores.append(score)

            per_model.setdefault(row.model_used, []).append(score)

        except Exception:
            continue

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
