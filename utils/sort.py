"""
-*- coding: utf-8 -*-
Name:        utils/process.py
Purpose:     Dataset preprocessing utilities (sorting, validation, de-dup, format standardization).
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from PIL import Image, ImageOps

from utils.config import JpegConfig, GENERATOR_CHOICES


@dataclass
class SortStats:
    scanned: int = 0
    moved: int = 0
    converted: int = 0
    invalid_resolution: int = 0
    duplicates: int = 0
    unreadable: int = 0

    # Optional bookkeeping for debugging / later features
    dup_examples: list[Tuple[Path, Path]] = field(default_factory=list)
    bad_res_examples: list[Path] = field(default_factory=list)
    unreadable_examples: list[Path] = field(default_factory=list)

    def render_summary(self) -> str:
        lines = [
            "=== sort summary ===",
            f"scanned:            {self.scanned}",
            f"moved:              {self.moved}",
            f"converted (png→jpg):{self.converted}",
            f"invalid resolution: {self.invalid_resolution}",
            f"duplicates:         {self.duplicates}",
            f"unreadable:         {self.unreadable}",
        ]
        return "\n".join(lines)


def _iter_images_ffhq(ffhq_root: Path) -> Iterable[Path]:
    """Expected: datasets/FFHQ/images1024x1024/*.png"""
    base = ffhq_root / "images1024x1024"
    if not base.exists():
        return []
    return (p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".png")


def _iter_images_tpdne(tpdne_root: Path) -> Iterable[Path]:
    """
    Expected: datasets/TPDNE/thispersondoesnotexist.10k/*.jpg
    Allow nested folders just in case.
    """
    if not tpdne_root.exists():
        return []
    return (
        p
        for p in tpdne_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )


def _iter_images_sfhq(sfhq_root: Path) -> Iterable[Path]:
    """Expected: datasets/SFHQ-T2I/images/images/*.jpg"""
    base = sfhq_root / "images" / "images"
    if not base.exists():
        return []
    return (
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )


def _sfhq_category_from_name(path: Path) -> Optional[str]:
    """Parse generator from `GEN_image_<number>` stem."""
    stem = path.stem  # e.g. FLUX1_pro_image_12345

    # Split from right: [generator, "image", number]
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None

    gen, middle, number = parts

    if middle != "image":
        return None

    if gen not in GENERATOR_CHOICES:
        return None

    # optional: ensure number is numeric
    if not number.isdigit():
        return None

    return gen


def _safe_move(src: Path, dst: Path) -> None:
    """
    Move efficiently. Path.rename() is best when same filesystem.
    Fall back to shutil.move otherwise.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        src.rename(dst)
    except OSError:
        shutil.move(str(src), str(dst))


def _content_hash_rgb1024(img: Image.Image) -> str:
    """
    Hash normalized pixel content (RGB). This ensures duplicates are detected
    across different encodings (png vs jpg) and locations.

    Note: This is an exact pixel hash after decoding; if two JPEGs differ
    slightly (even visually identical), they won't match. That’s usually what
    you want for "unique images" without perceptual hashing complexity.
    """
    # Ensure canonical orientation and mode
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Assumes already validated size
    raw = img.tobytes()  # ~3MB for 1024x1024 RGB
    h = hashlib.sha256()
    h.update(raw)
    return h.hexdigest()


def _open_image(path: Path) -> Optional[Image.Image]:
    """Open image or return None on failure."""
    try:
        # Use a context manager in caller when possible; here we return a live image object.
        img = Image.open(path)
        return img
    except Exception:
        return None


def sort_datasets(root: Path, jpeg: JpegConfig | None = None) -> SortStats:
    """
    Sort/preprocess datasets into:
      {root}/sorted/{FFHQ|TPDNE|FLUX1_schnell|FLUX1_pro|FLUX1_dev|SDXL|DALLE3}

    Workflow:
      Iterate each dataset -> check resolution -> check duplicate -> if pass,
      then convert filetype to jpg (FFHQ only) -> move to sorted bucket.

    Notes:
      - MOVE, do not copy
      - Leaves invalid/duplicates in place (only moves passing items)
    """
    root = Path(root)
    sorted_root = root / "sorted"
    sorted_root.mkdir(parents=True, exist_ok=True)

    ffhq_root = root / "FFHQ"
    sfhq_root = root / "SFHQ-T2I"
    tpdne_root = root / "TPDNE"

    seen_hashes: Set[str] = set()
    first_seen: Dict[str, Path] = {}

    stats = SortStats()
    jpeg = jpeg or JpegConfig()

    def handle_one(path: Path, category: str) -> None:
        nonlocal stats, seen_hashes, first_seen

        stats.scanned += 1

        img = _open_image(path)
        if img is None:
            stats.unreadable += 1
            if len(stats.unreadable_examples) < 10:
                stats.unreadable_examples.append(path)
            return

        try:
            if img.size != jpeg.standard_size:
                stats.invalid_resolution += 1
                if len(stats.bad_res_examples) < 10:
                    stats.bad_res_examples.append(path)
                return

            # De-dup based on decoded pixel content (before re-encoding)
            h = _content_hash_rgb1024(img)
            if h in seen_hashes:
                stats.duplicates += 1
                if len(stats.dup_examples) < 10:
                    stats.dup_examples.append((path, first_seen[h]))
                return

            seen_hashes.add(h)
            first_seen[h] = path

            dest_dir = sorted_root / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / f"{path.stem}.jpg"
            if dest_path.exists():
                dest_path = dest_dir / f"{path.stem}_{h[:8]}.jpg"

            # Re-encode EVERYTHING through the same JPEG pipeline
            rgb = img.convert("RGB")
            tmp_path = dest_path.with_suffix(".jpg.tmp")
            rgb.save(
                tmp_path,
                format="JPEG",
                quality=jpeg.quality,
                subsampling=jpeg.subsampling,
                optimize=jpeg.optimize,
                progressive=jpeg.progressive,
            )
            os.replace(tmp_path, dest_path)  # atomic when possible

            # Delete original after successful write (this is your “move”)
            path.unlink(missing_ok=True)

            stats.moved += 1
            # "converted" now means "re-encoded" (optional rename later)
            stats.converted += 1

        finally:
            try:
                img.close()
            except Exception:
                pass

    # Process FFHQ (real images)
    if ffhq_root.exists():
        for p in _iter_images_ffhq(ffhq_root):
            handle_one(p, category="FFHQ")

    # Process TPDNE (stylegan images)
    if tpdne_root.exists():
        for p in _iter_images_tpdne(tpdne_root):
            handle_one(p, category="TPDNE")

    # Process SFHQ-T2I (mixed)
    if sfhq_root.exists():
        for p in _iter_images_sfhq(sfhq_root):
            cat = _sfhq_category_from_name(p)
            if cat is None:
                # If it doesn't match naming convention, treat as unreadable/unknown bucket later
                # For now: skip with "unreadable" counter to keep stats obvious.
                stats.unreadable += 1
                if len(stats.unreadable_examples) < 10:
                    stats.unreadable_examples.append(p)
                continue
            handle_one(p, category=cat)

    return stats
