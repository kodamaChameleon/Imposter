"""
-*- coding: utf-8 -*-
Name:        run.py
Purpose:     Entry point for Imposter operations.
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
from utils import (
    make_config,
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
        res = run_kid(
            args.kid["path_a"],
            args.kid["path_b"],
            output=args.kid["output"],
            feature_model=args.feature_model,
        )

        print("[kid] dataset_a:", res.dataset_a)
        print("[kid] dataset_b:", res.dataset_b)
        print("[kid] num_images:", res.num_images_a, res.num_images_b)
        print("[kid] kid_mean:", res.kid_mean)
        print("[kid] kid_std :", res.kid_std)

        if args.kid["output"]:
            print("[ok] wrote:", args.kid["output"])

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
        input_root, output_root = args.transform

        run_transform(
            input_root=input_root,
            output_root=output_root,
            jpeg_cfg=configs.jpeg,
            opts=args.transform_opt,
            variations=args.transform_level[0],
            delta=args.transform_level[1],
        )

        print("[ok] transform complete")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
