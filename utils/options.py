"""
-*- coding: utf-8 -*-
Name:        utils/options.py
Purpose:     Handle runtime arguments
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import argparse
import sys
from pathlib import Path

from .config import DefaultOptions

DEFAULTS = DefaultOptions()


def build_parser() -> argparse.ArgumentParser:
    """
    Argument parser definitions
    """
    parser = argparse.ArgumentParser(
        description="Imposter Command-Line Utility"
    )

    # Download datasets
    parser.add_argument(
        "--download",
        nargs="+",
        choices=tuple(DEFAULTS.download_choices),
        help=f"Download datasets: {DEFAULTS.download_choices} (ex. --download ffhq tpdne).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULTS.root,
        help=f"Root datasets directory (default: {DEFAULTS.root}).",
    )

    # Data Processing
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort, process, and validate datasets.",
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
        default=DEFAULTS.feature_model,
        choices=DEFAULTS.feature_choices,
        help=f"Feature extractor for KID. (Default: {DEFAULTS.feature_model}). \
            Options: {', '.join(DEFAULTS.feature_choices)}",
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
        default=DEFAULTS.clip_mode,
        choices=DEFAULTS.clip_choices,
        help=f"CLIP scoring mode: sliding or truncate (default {DEFAULTS.clip_mode}).",
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
        default=DEFAULTS.trainval_set,
        help=f"Datasets to include in train/val splits \
            (default: {' '.join(DEFAULTS.trainval_set)})."
    )
    parser.add_argument(
        "--test-set",
        nargs="+",
        default=DEFAULTS.test_set,
        help=f"Datasets to include in test split (default: {' '.join(DEFAULTS.test_set)}).",
    )
    parser.add_argument(
        "--real-set",
        type=str,
        default=DEFAULTS.real_set,
        help=f"Name of real dataset (default: {DEFAULTS.real_set})."
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        default=DEFAULTS.split_ratios,
        help=f"Ratios for train/val/test splits \
            (default: {' '.join(map(str, DEFAULTS.split_ratios))}). Must sum to 1.0."
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULTS.split_csv,
        help=f"Output CSV path for splits (default: {DEFAULTS.split_csv}).")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULTS.split_seed,
        help=f"Random seed for splitting (default: {DEFAULTS.split_seed})."
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
        default=DEFAULTS.transform_opt,
        choices=DEFAULTS.transform_choices,
        help=f"Transform options (default: {' '.join(DEFAULTS.transform_opt)})."
    )

    parser.add_argument(
        "--transform-level",
        nargs=2,
        type=int,
        default=DEFAULTS.transform_level,
        metavar=("VARIATIONS", "DELTA"),
        help=f"Number of variations and decrement step \
            (default: {' '.join(map(str, DEFAULTS.transform_level))})."
    )

    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Validate provided arguments
    """

    # transform-level constraint
    x_var, x_delta = args.transform_level
    if x_var * x_delta > 80:
        raise ValueError(
            "Transform level cannot be greater than 80% "
            "(level = variations x delta)"
        )

    # Normalize "all"
    if args.download and "all" in args.download:
        args.download = list(
            set(DEFAULTS.download_choices) - {"all"}
        )

    if args.transform_opt and "all" in args.transform_opt:
        args.transform_opt = list(
            set(DEFAULTS.transform_choices) - {"all"}
        )

    # KID validation + coercion
    if args.kid:
        if len(args.kid) not in (2, 3):
            raise ValueError(
                "--kid expects 2 or 3 args: <pathA> <pathB> [output.json]"
            )

        args.kid = {
            "path_a": Path(args.kid[0]),
            "path_b": Path(args.kid[1]),
            "output": Path(args.kid[2]) if len(args.kid) == 3 else None,
        }

    # Transform
    if args.transform:
        if len(args.transform) != 2:
            raise ValueError(
                "--transform expects: <input_root> <output_root>"
            )

        args.transform = [
            Path(args.transform[0]),
            Path(args.transform[1]),
        ]

    # CLIP
    if args.clip is not None:
        # If a string is provided, wrap it as Path; else keep None
        if isinstance(args.clip, str):
            args.clip = Path(args.clip)
        else:
            args.clip = None

    return args


def parse_args(argv=None) -> argparse.Namespace | int:
    """
    Collect and validate user input
    """
    parser = build_parser()

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    args = parser.parse_args(argv)
    return validate_args(args)
