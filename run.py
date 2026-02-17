"""
-*- coding: utf-8 -*-
Name:        run.py
Purpose:     Entry point for DeepFail operations.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import argparse
import sys
from pathlib import Path

from utils.config import make_config, DOWNLOAD_CHOICES
from utils.download import fetch_all
from utils.sort import sort_datasets
from utils.kid import run_kid
from utils.clip import run_clip


def build_parser() -> argparse.ArgumentParser:
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # If no args were given, show help and exit 0
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return 0
    
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

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
