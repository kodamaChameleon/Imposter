"""
-*- coding: utf-8 -*-
Name:        run.py
Purpose:     Entry point for DeepFail operations.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
Co-Author:   ChatGPT 5.2
"""
import argparse
import sys
from pathlib import Path

from utils.config import make_config, DOWNLOAD_CHOICES
from utils.data import fetch_all

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepFail Command-Line Utility")

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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # If no args were given, show help and exit 0
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return 0
    
    # Download required datasets
    if len(args.download) > 0:

        # Normalize "all"
        if "all" in args.download:
            args.download = set(DOWNLOAD_CHOICES) - {"all"}

        cfg, kcfg = make_config(root=args.root)
        paths = fetch_all(cfg, kcfg, include=args.download)
        for name, p in paths.items():
            print(f"[ok] {name}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
