"""
-*- coding: utf-8 -*-
Name:        utils/kid.py
Purpose:     Measure KID distance between two datasets
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

from .config import IMG_EXTS


@dataclass
class KIDResult:
    """Structured output of a KID run (JSON-serializable)."""
    dataset_a: str
    dataset_b: str
    feature_model: str
    num_images_a: int
    num_images_b: int
    feature_dim: int
    subset_size: int
    n_subsets: int
    kid_mean: float
    kid_std: float
    seed: int


class ImageFolderRecursive(Dataset):
    """
    Recursively loads all images under `root` of valid `IMG_EXTS`.
    """
    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = Path(root)
        self.transform = transform
        self.files = self._gather_files(self.root)

        if len(self.files) == 0:
            raise ValueError(f"No images found under: {self.root}")

    @staticmethod
    def _gather_files(root: Path) -> List[Path]:
        """Return sorted list of image files under `root` (recursive)."""
        files: List[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
        files.sort()
        return files

    def __len__(self) -> int:
        """Number of discovered image files."""
        return len(self.files)

    def __getitem__(self, idx: int):
        """Load image → RGB → apply transform → return tensor."""
        path = self.files[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transform(im)
        return x


# ----------------------------
# Feature extractors
# ----------------------------

def _imagenet_norm():
    """
    Canonical ImageNet normalization
    (used by torchvision pretrained models)
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def build_preprocess(feature_model: str) -> transforms.Compose:
    """Return eval preprocessing for the given feature backbone."""
    mean, std = _imagenet_norm()

    if feature_model == "inception":
        # Inception-v3 expects 299x299
        return transforms.Compose([
            transforms.Resize(
                342,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if feature_model == "dinov2_vitb14":
        # Typically 518 gives stronger alignment with eval usage
        return transforms.Compose([
            transforms.Resize(
                584,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    raise ValueError(f"Unknown feature_model: {feature_model}")


class InceptionPool3(nn.Module):
    """Inception v3 that returns the 2048-dim pooled features (pool3)."""
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        # Your torchvision requires aux_logits=True when weights are set.
        model = inception_v3(
            weights=weights,
            aux_logits=True,
            transform_input=False
        )
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # (N, 2048)


class DinoV2ViTB14(nn.Module):
    """
    DINOv2 ViT-B/14 feature extractor.
    Returns the CLS-token embedding (dim typically 768).
    Loaded via torch.hub.
    """
    def __init__(self):
        super().__init__()
        # This will download weights on first run if not cached.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features that returns normalized tokens.
        feats = self.model.forward_features(x)
        # Common key in DINOv2: 'x_norm_clstoken'
        if "x_norm_clstoken" not in feats:
            raise RuntimeError(
                "Unexpected DINOv2 forward_features output: missing 'x_norm_clstoken'"
            )
        return feats["x_norm_clstoken"]  # (N, D)


def build_feature_extractor(feature_model: str) -> nn.Module:
    """Construct the requested feature backbone in eval mode."""
    if feature_model == "inception":
        return InceptionPool3()
    if feature_model == "dinov2_vitb14":
        return DinoV2ViTB14()
    raise ValueError(f"Unknown feature_model: {feature_model}")


@torch.no_grad()
def extract_features(
    image_root: Path,
    feature_model: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    max_images: Optional[int] = None,
) -> np.ndarray:
    """
    Returns: (N, D) float32 features
    Automatically caches features per dataset + feature model.
    """

    image_root = Path(image_root)
    dataset_name = image_root.name
    cache_path = image_root / f".features_{dataset_name}_{feature_model}.npz"

    # -------------------------
    # Load from cache if exists
    # -------------------------
    if cache_path.exists():
        print(f"[cache] loading features from {cache_path}")
        data = np.load(cache_path)
        return data["features"]

    print(f"[cache] computing features for {dataset_name} ({feature_model})")

    tfm = build_preprocess(feature_model)
    ds = ImageFolderRecursive(image_root, tfm)

    if max_images is not None:
        ds.files = ds.files[:max_images]

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=False,
    )

    model = build_feature_extractor(feature_model).to(device)
    model.eval()

    feats: List[np.ndarray] = []
    for batch in dl:
        batch = batch.to(device, non_blocking=True)
        f = model(batch)
        feats.append(f.detach().cpu().numpy().astype(np.float32))

    features = np.concatenate(feats, axis=0)

    # -------------------------
    # Save to cache
    # -------------------------
    print(f"[cache] saving features to {cache_path}")
    np.savez_compressed(cache_path, features=features)

    return features


# ----------------------------
# KID / MMD
# ----------------------------

def _polynomial_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    KID kernel: k(x,y) = ((x·y)/d + 1)^3
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    d = X.shape[1]
    K = (X @ Y.T) / float(d) + 1.0
    return K ** 3


def _mmd2_unbiased_polynomial(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD^2 estimator with polynomial kernel, for equal-sized sets.
    """
    m = X.shape[0]
    if m < 2:
        return float("nan")

    Kxx = _polynomial_kernel(X, X)
    Kyy = _polynomial_kernel(Y, Y)
    Kxy = _polynomial_kernel(X, Y)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()

    return float(term_xx + term_yy - 2.0 * term_xy)


def compute_kid(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    subset_size: int = 100,
    n_subsets: int = 100,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Estimate KID mean/std via subset sampling.

    Each subset uses `min(subset_size, len(A), len(B))` samples
    without replacement.
    """
    rng = np.random.default_rng(seed)

    na = feats_a.shape[0]
    nb = feats_b.shape[0]
    m = min(subset_size, na, nb)

    if m < 2:
        return float("nan"), float("nan")

    vals: List[float] = []
    for _ in range(n_subsets):
        ia = rng.choice(na, size=m, replace=False)
        ib = rng.choice(nb, size=m, replace=False)
        vals.append(_mmd2_unbiased_polynomial(feats_a[ia], feats_b[ib]))

    vals_np = np.array(vals, dtype=np.float64)
    kid_mean = float(vals_np.mean())
    kid_std = float(vals_np.std(ddof=1)) if len(vals) > 1 else 0.0
    return kid_mean, kid_std


def run_kid(
    path_a: Path,
    path_b: Path,
    output: Optional[Path] = None,
    feature_model: str = "inception",
    device: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_size: int = 100,
    n_subsets: int = 100,
    seed: int = 0,
    max_images: Optional[int] = None,
) -> KIDResult:
    """
    End-to-end KID between two image directories.

    - Extracts (or loads cached) features
    - Computes subset KID
    - Optionally writes JSON result

    Device defaults to CUDA when available.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    feats_a = extract_features(
        path_a,
        feature_model=feature_model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
    )
    feats_b = extract_features(
        path_b,
        feature_model=feature_model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
    )

    kid_mean, kid_std = compute_kid(
        feats_a, feats_b, subset_size=subset_size, n_subsets=n_subsets, seed=seed
    )

    result = KIDResult(
        dataset_a=str(path_a),
        dataset_b=str(path_b),
        feature_model=feature_model,
        num_images_a=int(feats_a.shape[0]),
        num_images_b=int(feats_b.shape[0]),
        feature_dim=int(feats_a.shape[1]),
        subset_size=int(min(subset_size, feats_a.shape[0], feats_b.shape[0])),
        n_subsets=int(n_subsets),
        kid_mean=float(kid_mean),
        kid_std=float(kid_std),
        seed=int(seed),
    )

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    return result
