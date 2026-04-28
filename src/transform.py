"""
-*- coding: utf-8 -*-
Name:        utils/transform.py
Purpose:     Dataset transformation utilities.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
from typing import Iterable
import csv

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import lpips
import torchvision.transforms as T

from .config import IMG_EXTS


# ------------------------------------------------------------------
# LPIPS
# ------------------------------------------------------------------

class LPIPSScorer:
    """
    Standard metric for cross feature comparison
    - device: Uses cuda by default if available
    - net: Alex from the original LPIPS paper should suffice
        Other options include "squeeze" and "vgg"
    """
    def __init__(self, device=None, net="alex"):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

        self.tf = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def tensorize(self, img):
        return self.tf(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def score_batch(self, refs, imgs):

        refs = torch.cat(refs)
        imgs = torch.cat(imgs)

        scores = self.model(refs, imgs)

        assert refs.device == imgs.device

        return scores.flatten().cpu().tolist()


# ------------------------------------------------------------------
# Core class
# ------------------------------------------------------------------

class DatasetTransformer:
    """
    Transform a dataset by specified:
    - Levels
    - Delta
    - Feature type
    """

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        jpeg_cfg,
        report_path: Path,
        scorer: LPIPSScorer | None = None
    ):
        self.input_root = input_root.resolve()
        self.output_root = output_root.resolve()
        self.dataset_name = self.input_root.name
        self.jpeg_cfg = jpeg_cfg
        self.report_path = report_path.resolve()

        self.images = tuple(self._iter_images())
        self.scorer = scorer or LPIPSScorer()

        self.transforms = {
            "compression": (self._levels_decreasing, self._tf_compression),
            "resize": (self._levels_decreasing, self._tf_resize),
            "crop": (self._levels_decreasing, self._tf_crop),
            "contrast": (self._levels_symmetric, self._tf_contrast),
            "saturation": (self._levels_symmetric, self._tf_saturation),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, opts, variations: int, delta: int, start: int):
        """
        Initialize a transformation run
        """

        if not self.images:
            print("[warn] no images found")
            return

        for name in opts:
            level_fn, tf_fn = self.transforms[name]

            levels = level_fn(variations, delta, start)

            self._run_levels(name, levels, tf_fn)

    # ------------------------------------------------------------------
    # Execution engine
    # ------------------------------------------------------------------

    def _run_levels(self, name, levels, transform_fn, batch_size: int = 32):
        """
        Generic wrapper for running transformations
        """

        for level in levels:
            
            ref_batch = []
            img_batch = []

            lpips_sum = 0.0
            count = 0

            out_dir = self.output_root / f"{self.dataset_name}-{name}-{level}"

            for img_path in tqdm(self.images, desc=f"[{name} {level}]"):

                rel = img_path.relative_to(self.input_root)
                out_path = out_dir / rel

                try:
                    with Image.open(img_path) as img:

                        img = img.convert("RGB")
                        out_img, quality = transform_fn(img, level)

                        self._save_jpeg(out_img, out_path, quality)

                        with Image.open(out_path.with_suffix(".jpg")) as comp_img:
                            comp_img = comp_img.convert("RGB")

                            ref_batch.append(self.scorer.tensorize(img))
                            img_batch.append(self.scorer.tensorize(comp_img))

                        if len(ref_batch) == batch_size:

                            scores = self.scorer.score_batch(ref_batch, img_batch)

                            for s in scores:
                                lpips_sum += s
                                count += 1

                            ref_batch.clear()
                            img_batch.clear()

                except Exception as e:
                    print(f"[skip] {img_path}: {e}")
                
            if ref_batch:
                scores = self.scorer.score_batch(ref_batch, img_batch)

                for s in scores:
                    lpips_sum += s
                    count += 1

            mean_lpips = lpips_sum / count if count else 0.0

            # Print status message
            print(
                f"[done] mean_lpips={mean_lpips}"
            )

            self._append_report({
                "dataset": self.dataset_name,
                "transform": name,
                "level": level,
                "lpips": mean_lpips,
                "images": count,
            })

    # ------------------------------------------------------------------
    # Transform implementations
    # ------------------------------------------------------------------

    def _tf_compression(self, img, level):
        """
        Re-encode images at varying JPEG quality levels.
        """
        q = max(1, min(level, 100))
        return img, q

    def _tf_resize(self, img, level):
        """
        Vary images by percentage levels and re-encode as JPEG.
        """
        w, h = img.size
        new_size = (max(1, int(w * level / 100)), max(1, int(h * level / 100)))
        return img.resize(new_size, Image.LANCZOS), self.jpeg_cfg.quality

    def _tf_crop(self, img, level):
        """
        Center-crop images by percentage and re-encode as JPEG.
        """
        w, h = img.size
        keep = level / 100

        new_w = max(1, int(w * keep))
        new_h = max(1, int(h * keep))

        left = (w - new_w) // 2
        top = (h - new_h) // 2

        return img.crop((left, top, left + new_w, top + new_h)), self.jpeg_cfg.quality

    def _tf_contrast(self, img, level):
        """
        Apply symmetric contrast variations and re-encode as JPEG.
        """
        return self._apply_contrast(img, level / 100.0), self.jpeg_cfg.quality

    def _tf_saturation(self, img, level):
        """
        Apply symmetric saturation variations and re-encode as JPEG.
        """
        return self._apply_saturation(img, level / 100.0), self.jpeg_cfg.quality

    # ------------------------------------------------------------------
    # Level generators
    # ------------------------------------------------------------------

    def _levels_decreasing(self, variations, delta, start):
        """
        Generate decreasing levels starting from `start`.
        """
        levels = []
        for i in range(variations):
            lvl = start - (i + 1) * delta
            if lvl <= 0:
                break
            levels.append(lvl)
        return levels

    def _levels_symmetric(self, variations, delta, start):
        """
        Return sorted levels symmetric around `start`.
        """
        levels = []

        for i in range(1, variations + 1):

            down = start - i * delta
            up = start + i * delta

            if down > 0:
                levels.append(down)

            levels.append(up)

        return sorted(levels)

    # ------------------------------------------------------------------
    # Image ops
    # ------------------------------------------------------------------

    def _apply_contrast(self, img, factor):
        """Apply linear contrast in normalized RGB space."""
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) * factor + 0.5
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _apply_saturation(self, img, factor):
        """Scale color distance from luminance (Rec. 709)."""
        arr = np.asarray(img).astype(np.float32) / 255.0

        lum = (
            0.2126 * arr[..., 0]
            + 0.7152 * arr[..., 1]
            + 0.0722 * arr[..., 2]
        )[..., None]

        arr = lum + factor * (arr - lum)
        arr = np.clip(arr, 0, 1)

        return Image.fromarray((arr * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    def _iter_images(self) -> Iterable[Path]:
        """Yield image paths under `root` whose suffix is in `IMG_EXTS`."""
        for p in self.input_root.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                yield p

    def _save_jpeg(self, img, out_path, quality):
        """Save image as JPEG to `out_path`."""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img.save(
            out_path.with_suffix(".jpg"),
            format="JPEG",
            quality=quality,
            subsampling=self.jpeg_cfg.subsampling,
            optimize=self.jpeg_cfg.optimize,
            progressive=self.jpeg_cfg.progressive,
        )

    def _append_report(self, row: dict):
        """Record results"""

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.report_path.exists()

        with self.report_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_transform(
    input_root: Path,
    output_root: Path,
    jpeg_cfg,
    opts,
    variations: int,
    delta: int,
    start: int,
    report_path: str | Path,
):
    """
    CLI wrapper for easy execution
    """
    DatasetTransformer(
        input_root,
        output_root,
        jpeg_cfg,
        Path(report_path),
    ).run(opts, variations, delta, start)
