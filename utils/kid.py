"""
-*- coding: utf-8 -*-
Name:        utils/validation.py
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


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class KIDResult:
    dataset_a: str
    dataset_b: str
    num_images_a: int
    num_images_b: int
    feature_dim: int
    subset_size: int
    n_subsets: int
    kid_mean: float
    kid_std: float
    seed: int


class ImageFolderRecursive(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = Path(root)
        self.transform = transform
        self.files = self._gather_files(self.root)

        if len(self.files) == 0:
            raise ValueError(f"No images found under: {self.root}")

    @staticmethod
    def _gather_files(root: Path) -> List[Path]:
        files: List[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
        files.sort()
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        # Always convert to RGB for inception
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transform(im)
        return x


class InceptionPool3(nn.Module):
    """
    Inception v3 that returns the 2048-dim pooled features (pool3).
    """
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        model.eval()
        # We only need up through the final avgpool
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This matches torchvision inception forward blocks, stopping at avgpool.
        # Input: (N, 3, 299, 299)
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

        # Adaptive avgpool to (1,1), then flatten -> (N, 2048)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def _inception_preprocess() -> transforms.Compose:
    # Stable ImageNet normalization used across torchvision pretrained ImageNet models
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.no_grad()
def extract_inception_features(
        image_root: Path,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        max_images: Optional[int] = None,
    ) -> np.ndarray:
    """
    Returns: (N, 2048) float32 features
    """
    image_root = Path(image_root)
    tfm = _inception_preprocess()
    ds = ImageFolderRecursive(image_root, tfm)

    if max_images is not None:
        # simple truncation for debugging
        ds.files = ds.files[:max_images]

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=False,
    )

    model = InceptionPool3().to(device)
    model.eval()

    feats: List[np.ndarray] = []
    for batch in dl:
        batch = batch.to(device, non_blocking=True)
        f = model(batch)  # (B, 2048)
        feats.append(f.detach().cpu().numpy().astype(np.float32))

    out = np.concatenate(feats, axis=0)
    return out


def _polynomial_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    KID kernel: k(x,y) = ((x·y)/d + 1)^3
    X: (m, d), Y: (n, d)
    returns: (m, n)
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    d = X.shape[1]
    K = (X @ Y.T) / float(d) + 1.0
    return K ** 3


def _mmd2_unbiased_polynomial(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD^2 estimator with polynomial kernel, for equal-sized sets.
    X, Y: (m, d)
    """
    m = X.shape[0]
    if m < 2:
        return float("nan")

    Kxx = _polynomial_kernel(X, X)
    Kyy = _polynomial_kernel(Y, Y)
    Kxy = _polynomial_kernel(X, Y)

    # remove diagonal terms for unbiased estimate
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()  # already includes all pairs

    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return float(mmd2)


def compute_kid(
        feats_a: np.ndarray,
        feats_b: np.ndarray,
        subset_size: int = 100,
        n_subsets: int = 100,
        seed: int = 0,
    ) -> Tuple[float, float]:
    """
    KID is the mean of MMD^2 over random subsets.
    Returns: (mean, std)
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
        Xa = feats_a[ia]
        Xb = feats_b[ib]
        vals.append(_mmd2_unbiased_polynomial(Xa, Xb))

    vals_np = np.array(vals, dtype=np.float64)
    return float(vals_np.mean()), float(vals_np.std(ddof=1)) if len(vals) > 1 else 0.0


def run_kid(
        path_a: Path,
        path_b: Path,
        output: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        subset_size: int = 100,
        n_subsets: int = 100,
        seed: int = 0,
        max_images: Optional[int] = None,
    ) -> KIDResult:
    path_a = Path(path_a)
    path_b = Path(path_b)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    feats_a = extract_inception_features(
        path_a, device=device, batch_size=batch_size, num_workers=num_workers, max_images=max_images
    )
    feats_b = extract_inception_features(
        path_b, device=device, batch_size=batch_size, num_workers=num_workers, max_images=max_images
    )

    kid_mean, kid_std = compute_kid(
        feats_a, feats_b, subset_size=subset_size, n_subsets=n_subsets, seed=seed
    )

    result = KIDResult(
        dataset_a=str(path_a),
        dataset_b=str(path_b),
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
