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

    # Sort / preprocess datasets
    if args.sort:
        stats = sort_datasets(root=args.root, jpeg=jcfg)
        print(stats.render_summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
