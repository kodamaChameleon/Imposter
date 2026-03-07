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
        help="Compute KID: --kid <pathA> <pathB> <pathC> ..."
    )
    parser.add_argument(
        "--kid-results",
        type=Path,
        default=DEFAULTS.kid_csv,
        help="CSV file to store KID results (append mode).",
    )
    parser.add_argument(
        "--feature-model",
        type=str,
        default=DEFAULTS.kid_model,
        choices=DEFAULTS.kid_choices,
        help=f"Feature extractor for KID. (Default: {DEFAULTS.kid_model}). \
            Options: {', '.join(DEFAULTS.kid_choices)}",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Compute CLIP score for SFHQ-T2I.",
    )
    parser.add_argument(
        "--clip-results",
        type=Path,
        default=DEFAULTS.clip_csv,
        help="CSV file to store CLIP results (append mode).",
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
    parser.add_argument(
    "--transform-report",
        default=DEFAULTS.transform_csv,
        help=f"CSV file to store average LPIPS scores per transform level \
            (default: {DEFAULTS.transform_csv})."
    )

    parser.add_argument("--graph", type=Path)
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=DEFAULTS.graph_choices,
        help=f"Data type of results file \
            (OPTIONS: {' '.join(DEFAULTS.graph_choices)})"
    )
    parser.add_argument(
        "--normalize",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="Path to results data to normalize (<pathA> <pathB> <pathC> ...)"
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=DEFAULTS.normalized_output,
        help="Path to store normalization output.",
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
        if len(args.kid) < 2:
            raise ValueError(
                "--kid expects: <pathA> <pathB> <pathC> ..."
            )

        args.kid = {
            "path_a": Path(args.kid[0]),
            "path_bs": [Path(p) for p in args.kid[1:]],
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
