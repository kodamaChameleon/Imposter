"""
-*- coding: utf-8 -*-
Name:        run.py
Purpose:     Entry point for Imposter operations.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from utils import (
    make_config,
    SplitConfig,
    fetch_all,
    sort_datasets,
    run_kid,
    run_clip,
    run_split,
    run_transform,
    parse_args,
)


def main() -> int:
    """
    Build configs and execute based on the provided arguements
    """
    args = parse_args()
    configs = make_config(root=args.root)

    if args.download and len(args.download) > 0:

        paths = fetch_all(configs.dataset, configs.kaggle, include=args.download)
        for name, p in paths.items():
            print(f"[ok] {name}: {p}")

    if args.sort:
        stats = sort_datasets(root=args.root, jpeg=configs.jpeg)
        print(stats.render_summary())

    if args.kid:
        results = run_kid(
            path_a=args.kid["path_a"],
            path_b=args.kid["path_bs"],
            csv_path=args.kid_results,
            feature_model=args.feature_model,
        )

        for r in results:
            print(
                f"[kid] {r.dataset_a} vs {r.dataset_b} | "
                f"n={r.num_images_a}/{r.num_images_b} | "
                f"KID={r.kid_mean:.6f} ± {r.kid_std:.6f}"
            )

        if args.kid_results:
            print(f"[ok] appended → {args.kid_results}")

    if args.clip is not None:
        res = run_clip(root=args.root, output=args.clip)

        print("[clip] dataset:", res.dataset)
        print("[clip] num_images:", res.num_images)
        print("[clip] global_mean:", res.global_mean)

        for m, v in res.per_model.items():
            print(f"[clip] {m}: {v}")

        if args.clip:
            print("[ok] wrote:", args.clip)

    if args.split:
        split_cfg = SplitConfig(
            root=args.root,
            trainval_sources=args.trainval_set,
            test_sources=args.test_set,
            real_source=args.real_set,
            ratios=tuple(args.split_ratios),
            csv_path=args.split_csv or args.root / "train_val_test.csv",
            seed=args.split_seed,
        )

        run_split(split_cfg)

        print(f"[ok] split written to {split_cfg.csv_path}")

    if args.transform:
        input_root, output_root = args.transform

        run_transform(
            input_root=input_root,
            output_root=output_root,
            jpeg_cfg=configs.jpeg,
            opts=args.transform_opt,
            variations=args.transform_level[0],
            delta=args.transform_level[1],
            report_path=args.transform_report,
        )

        print("[ok] transform complete")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
