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

from utils.config import DATASETS
from utils.data import provision_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repo runner / dataset provisioner.")

    parser.add_argument(
        "--sort",
        nargs="*",
        choices=["train", "val", "test", "all"],
        default=[],
        help=(
            "Which split(s) to provision/sort. "
            "Choose from: train, val, test. "
            "You can pass multiple, e.g. --sort train val. "
        ),
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

    # Sort data
    if len(args.sort) > 0:
        if "all" in args.sort:
            args.sort = ["train", "val", "test"]
        provision_datasets(DATASETS, root=args.root, splits=args.sort)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
