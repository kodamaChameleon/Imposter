"""
-*- coding: utf-8 -*-
Name:        utils/transform.py
Purpose:     Dataset transformation utilities.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import IMG_EXTS


def _iter_images(root: Path) -> Iterable[Path]:
    """Yield image paths under `root` whose suffix is in `IMG_EXTS`."""
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def _save_jpeg(
    img: Image.Image,
    out_path: Path,
    quality: int,
    jpeg_cfg
):
    """Save image as JPEG to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = img.convert("RGB")
    img.save(
        out_path.with_suffix(".jpg"),
        format="JPEG",
        quality=quality,
        subsampling=jpeg_cfg.subsampling,
        optimize=jpeg_cfg.optimize,
        progressive=jpeg_cfg.progressive,
    )


def _symmetric_levels(variations: int, delta: int) -> list[int]:
    """
    Return sorted percentage levels symetric around 100
    where 100 represents original image state.
    """
    levels = []
    for i in range(1, variations + 1):
        down = 100 - i * delta
        up = 100 + i * delta

        if down > 0:
            levels.append(down)

        levels.append(up)

    return sorted(levels)


def _apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Apply linear contrast in normalized RGB space."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) * factor + 0.5
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _apply_saturation(img: Image.Image, factor: float) -> Image.Image:
    """Scale color distance from luminance (Rec. 709)."""
    arr = np.asarray(img).astype(np.float32) / 255.0

    # perceptual luminance (Rec. 709)
    lum = (
        0.2126 * arr[..., 0]
        + 0.7152 * arr[..., 1]
        + 0.0722 * arr[..., 2]
    )[..., None]

    arr = lum + factor * (arr - lum)
    arr = np.clip(arr, 0, 1)

    return Image.fromarray((arr * 255).astype(np.uint8))


def _compression_levels(
    base_quality: int,
    variations: int,
    delta: int
) -> list[int]:
    """Generate decreasing JPEG quality levels from `base_quality`."""
    levels = []
    for i in range(variations):
        q = base_quality - (i + 1) * delta
        if q <= 0:
            break
        levels.append(q)
    return levels


def _run_compression(
    images,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    jpeg_cfg,
    variations: int,
    delta: int,
):
    """
    Re-encode images at lower JPEG quality levels.
    """
    levels = _compression_levels(jpeg_cfg.quality, variations, delta)

    for q in levels:
        out_dir = output_root / f"{dataset_name}-compression-{q}"

        for img_path in tqdm(images, desc=f"[compression {q}]"):
            rel = img_path.relative_to(input_root)
            out_path = out_dir / rel

            try:
                with Image.open(img_path) as img:
                    _save_jpeg(img, out_path, q, jpeg_cfg)
            except Exception as e:
                print(f"[skip] {img_path}: {e}")


def _run_resize(
    images,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    variations: int,
    delta: int,
    jpeg_cfg,
):
    """
    Downscale images by percentage levels and re-encode as JPEG.
    """
    levels = [100 - (i + 1) * delta for i in range(variations) if 100 - (i + 1) * delta > 0]

    for scale in levels:
        out_dir = output_root / f"{dataset_name}-resize-{scale}"

        for img_path in tqdm(images, desc=f"[resize {scale}%]"):
            rel = img_path.relative_to(input_root)
            out_path = out_dir / rel

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    new_size = (
                        max(1, int(w * scale / 100)),
                        max(1, int(h * scale / 100)),
                    )

                    resized = img.resize(new_size, Image.LANCZOS)

                    _save_jpeg(resized, out_path, jpeg_cfg.quality, jpeg_cfg)

            except Exception as e:
                print(f"[skip] {img_path}: {e}")


def _run_crop(
    images,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    variations: int,
    delta: int,
    jpeg_cfg,
):
    """
    Center-crop images by percentage and re-encode as JPEG.
    """
    levels = [(i + 1) * delta for i in range(variations) if (i + 1) * delta < 100]

    for crop_pct in levels:
        out_dir = output_root / f"{dataset_name}-crop-{crop_pct}"

        for img_path in tqdm(images, desc=f"[crop {crop_pct}%]"):
            rel = img_path.relative_to(input_root)
            out_path = out_dir / rel

            try:
                with Image.open(img_path) as img:
                    w, h = img.size

                    keep_ratio = 1 - crop_pct / 100
                    new_w = max(1, int(w * keep_ratio))
                    new_h = max(1, int(h * keep_ratio))

                    left = (w - new_w) // 2
                    top = (h - new_h) // 2
                    right = left + new_w
                    bottom = top + new_h

                    cropped = img.crop((left, top, right, bottom))

                    _save_jpeg(cropped, out_path, jpeg_cfg.quality, jpeg_cfg)

            except Exception as e:
                print(f"[skip] {img_path}: {e}")


def _run_contrast(
    images,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    variations: int,
    delta: int,
    jpeg_cfg,
):
    """
    Apply symmetric contrast variations and re-encode as JPEG.
    """
    levels = _symmetric_levels(variations, delta)

    for level in levels:
        factor = level / 100.0
        out_dir = output_root / f"{dataset_name}-contrast-{level}"

        for img_path in tqdm(images, desc=f"[contrast {level}]"):
            rel = img_path.relative_to(input_root)
            out_path = out_dir / rel

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    out = _apply_contrast(img, factor)
                    _save_jpeg(out, out_path, jpeg_cfg.quality, jpeg_cfg)
            except Exception as e:
                print(f"[skip] {img_path}: {e}")


def _run_saturation(
    images,
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    variations: int,
    delta: int,
    jpeg_cfg,
):
    """
    Apply symmetric saturation variations and re-encode as JPEG.
    """
    levels = _symmetric_levels(variations, delta)

    for level in levels:
        factor = level / 100.0
        out_dir = output_root / f"{dataset_name}-saturation-{level}"

        for img_path in tqdm(images, desc=f"[saturation {level}]"):
            rel = img_path.relative_to(input_root)
            out_path = out_dir / rel

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    out = _apply_saturation(img, factor)
                    _save_jpeg(out, out_path, jpeg_cfg.quality, jpeg_cfg)
            except Exception as e:
                print(f"[skip] {img_path}: {e}")


def run_transform(
    input_root: Path,
    output_root: Path,
    jpeg_cfg,
    opts,
    variations: int,
    delta: int,
):
    """
    Execute selected dataset transformations.

    Parameters
    ----------
    input_root: Source dataset directory.
    output_root: Destination root for transformed datasets.
    jpeg_cfg: JPEG encoding configuration.
    opts: Iterable of transform names
    variations: Number of levels per transform.
    delta: Step size between levels (percentage).
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    dataset_name = input_root.name

    images = list(_iter_images(input_root))
    if not images:
        print("[warn] no images found")
        return

    if "compression" in opts:
        _run_compression(
            images,
            input_root,
            output_root,
            dataset_name,
            jpeg_cfg,
            variations,
            delta,
        )

    if "resize" in opts:
        _run_resize(
            images,
            input_root,
            output_root,
            dataset_name,
            variations,
            delta,
            jpeg_cfg,
        )

    if "crop" in opts:
        _run_crop(
            images,
            input_root,
            output_root,
            dataset_name,
            variations,
            delta,
            jpeg_cfg,
        )

    if "contrast" in opts:
        _run_contrast(
            images,
            input_root,
            output_root,
            dataset_name,
            variations,
            delta,
            jpeg_cfg,
        )

    if "saturation" in opts:
        _run_saturation(
            images,
            input_root,
            output_root,
            dataset_name,
            variations,
            delta,
            jpeg_cfg,
        )
