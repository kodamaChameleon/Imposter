"""
-*- coding: utf-8 -*-
Name:        run.py
Purpose:     Entry point for DeepFail operations.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import argparse
import sys
from pathlib import Path

from utils.config import make_config, DOWNLOAD_CHOICES, TRANSFORM_CHOICES
from utils.download import fetch_all
from utils.sort import sort_datasets
from utils.kid import run_kid
from utils.clip import run_clip
from utils.split import run_split
from utils.transform import run_transform


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepFail Command-Line Utility")

    # Download datasets
    parser.add_argument(
        "--download",
        nargs="+",
        choices=DOWNLOAD_CHOICES,
        help="Download datasets: all, sfhq_t2i, ffhq, tdpne (ex. --download ffhq tpdne).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Root datasets directory (default: datasets).",
    )

    # Data Processing
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort/preprocess datasets into datasets/sorted (validate 1024x1024, de-dup across all, FFHQ png->jpg, move files).",
    )
    parser.add_argument(
        "--kid",
        nargs="+",
        metavar="PATH",
        help="Compute KID between two datasets. Usage: --kid <pathA> <pathB> [output.json]",
    )
    parser.add_argument(
        "--feature-model",
        type=str,
        default="inception",
        choices=["inception", "dinov2_vitb14"],
        help="Feature extractor for KID. Default: inception (ImageNet). Options: inception, dinov2_vitb14",
    )
    parser.add_argument(
    "--clip",
        nargs="?",
        const="clip.json",
        metavar="OUTPUT",
        help="Compute CLIP score for SFHQ-T2I (optionally provide output.json).",
    )
    parser.add_argument(
        "--clip-mode",
        type=str,
        default="sliding",
        choices=["sliding", "truncate"],
        help="CLIP scoring mode: sliding (long prompts) or truncate (standard 77 token).",
    )

    # Split datasets into train/val/test
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create train/val/test splits."
    )
    parser.add_argument(
        "--trainval-set",
        nargs="+",
        default=["TPDNE"],
        help="Datasets to include in train/val splits (default: TPDNE)."
    )
    parser.add_argument(
        "--test-set",
        nargs="+",
        default=["FLUX1_dev", "FLUX1_pro", "FLUX1_schnell", "SDXL", "TPDNE"],
        help="Datasets to include in test split (default: FLUX1_dev FLUX1_pro FLUX1_schnell SDXL TPDNE).",
    )
    parser.add_argument(
        "--real-set",
        type=str,
        default="FFHQ",
        help="Name of real dataset (default: FFHQ)."
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        default=[0.6, 0.2, 0.2],
        help="Ratios for train/val/test splits (default: 0.6 0.2 0.2). Must sum to 1.0."
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default="datasets/train_val_test.csv",
        help="Output CSV path for splits (default: datasets/train_val_test.csv).")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1337,
        help="Random seed for splitting (default: 1337)."
    )

    # Transform datasets
    parser.add_argument(
        "--transform",
        nargs=2,
        metavar=("INPUT_ROOT", "OUTPUT_ROOT"),
        help="Apply dataset transformations."
    )

    parser.add_argument(
        "--transform-opt",
        nargs="+",
        default=["all"],
        choices=TRANSFORM_CHOICES,
        help="Transform options (default: all)."
    )

    parser.add_argument(
        "--transform-level",
        nargs=2,
        type=int,
        default=[6, 10],
        metavar=("VARIATIONS", "DELTA"),
        help="Number of variations and decrement step (default: 6 10)."
    )

    # Validation
    args = parser.parse_args()

    # If no args were given, show help and exit 0
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return 0

    x_sets, x_delta = args.transform_level
    if x_sets*x_delta > 80:
        raise ValueError("Transform level cannot be greater than 80% (level = sets x delta)")
    
    return args


def main() -> int:
    args = parse_args()

    # Make config
    cfg, kcfg, jcfg = make_config(root=args.root)

    # Download required datasets
    if args.download and len(args.download) > 0:

        # Normalize "all"
        if "all" in args.download:
            args.download = set(DOWNLOAD_CHOICES) - {"all"}

        
        paths = fetch_all(cfg, kcfg, include=args.download)
        for name, p in paths.items():
            print(f"[ok] {name}: {p}")

    # Sorting and preprocess datasets
    if args.sort:
        stats = sort_datasets(root=args.root, jpeg=jcfg)
        print(stats.render_summary())

    if args.kid:
        if len(args.kid) not in (2, 3):
            print("[err] --kid expects 2 or 3 args: <pathA> <pathB> [output.json]")
            return 2

        path_a = Path(args.kid[0])
        path_b = Path(args.kid[1])
        output = Path(args.kid[2]) if len(args.kid) == 3 else None

        res = run_kid(path_a, path_b, output=output, feature_model=args.feature_model)
        print("[kid] dataset_a:", res.dataset_a)
        print("[kid] dataset_b:", res.dataset_b)
        print("[kid] num_images:", res.num_images_a, res.num_images_b)
        print("[kid] kid_mean:", res.kid_mean)
        print("[kid] kid_std :", res.kid_std)
        if output is not None:
            print("[ok] wrote:", output)
    
    if args.clip is not None:
        output = Path(args.clip) if isinstance(args.clip, str) else None

        res = run_clip(root=args.root, output=output)

        print("[clip] dataset:", res.dataset)
        print("[clip] num_images:", res.num_images)
        print("[clip] global_mean:", res.global_mean)

        for m, v in res.per_model.items():
            print(f"[clip] {m}: {v}")

        if output:
            print("[ok] wrote:", output)
    
    # Split and transform
    if args.split:
        csv_path = args.split_csv or args.root / "train_val_test.csv"

        run_split(
            root=args.root,
            trainval_sources=args.trainval_set,
            test_sources=args.test_set,
            real_source=args.real_set,
            ratios=tuple(args.split_ratios),
            csv_path=csv_path,
            seed=args.split_seed,
        )

        print(f"[ok] split written to {csv_path}")

    if args.transform:
        input_root = Path(args.transform[0])
        output_root = Path(args.transform[1])

        if "all" in args.transform_opt:
            args.transform_opt = set(TRANSFORM_CHOICES) - {"all"}

        run_transform(
            input_root=input_root,
            output_root=output_root,
            jpeg_cfg=jcfg,
            opts=args.transform_opt,
            variations=args.transform_level[0],
            delta=args.transform_level[1],
        )

        print("[ok] transform complete")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
