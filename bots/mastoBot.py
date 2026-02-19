"""
-*- coding: utf-8 -*-
Name:        mastoBot.py
Purpose:     Filter images through a Mastodon upload/post cycle
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""

import argparse
import os
import time
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()
MAX_BATCH = 4
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------- helpers ----------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path")
    parser.add_argument("--dir_path")

    parser.add_argument("--text", default="Posted from Python 🐘")
    parser.add_argument("--out", default="datasets/results")

    parser.add_argument("--base-url", default=os.getenv("MASTODON_BASE_URL"))
    parser.add_argument("--token", default=os.getenv("MASTODON_ACCESS_TOKEN"))

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--delay", type=int, default=360)

    parser.add_argument("--delete", action="store_true")

    args = parser.parse_args()

    if not args.image_path and not args.dir_path:
        raise SystemExit("Provide --image_path or --dir_path")

    if args.image_path and args.dir_path:
        raise SystemExit("Only one input source allowed")

    if not args.base_url or not args.token:
        raise SystemExit("Missing Mastodon credentials")

    args.batch_size = min(args.batch_size, MAX_BATCH)

    os.makedirs(args.out, exist_ok=True)

    return args


def collect_images(args) -> List[Path]:
    if args.image_path:
        return [Path(args.image_path)]

    return sorted([
        p for p in Path(args.dir_path).iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])


# ---------- mastodon core ----------

def upload_media(base_url, token, path):
    url = f"{base_url}/api/v2/media"

    with open(path, "rb") as f:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            files={"file": f},
            timeout=60
        )

    r.raise_for_status()
    return r.json()


def create_status(base_url, token, text, media_ids):
    url = f"{base_url}/api/v1/statuses"

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        data={
            "status": text,
            "media_ids[]": media_ids
        },
        timeout=60
    )

    r.raise_for_status()
    return r.json()


def download_variants(media, src_path, out_dir):
    stem = src_path.stem
    ext = src_path.suffix

    for key in ("url", "preview_url"):
        if not media.get(key):
            continue

        tag = "original" if key == "url" else "small"

        r = requests.get(media[key], timeout=60)
        r.raise_for_status()

        out_path = Path(out_dir) / f"{stem}_{tag}{ext}"

        with open(out_path, "wb") as f:
            f.write(r.content)

    print(f"Downloaded: {stem}")


def delete_status(base_url, token, status_id):
    requests.delete(
        f"{base_url}/api/v1/statuses/{status_id}",
        headers={"Authorization": f"Bearer {token}"}
    )


# ---------- batching ----------

def process_batches(images, args):

    for i in range(0, len(images), args.batch_size):

        batch = images[i:i + args.batch_size]

        media_ids = []
        media_meta = []

        for path in batch:
            media = upload_media(args.base_url, args.token, path)
            media_ids.append(media["id"])
            media_meta.append(media)

        status = create_status(
            args.base_url,
            args.token,
            args.text,
            media_ids
        )

        print(f"Posted: {status['url']}")

        for media, src in zip(media_meta, batch):
            download_variants(media, src, args.out)

        if args.delete:
            delete_status(args.base_url, args.token, status["id"])
            print("Deleted post")

        if i + args.batch_size < len(images):
            print(f"Sleeping {args.delay}s...")
            time.sleep(args.delay)


# ---------- main ----------

def main():
    args = parse_args()
    images = collect_images(args)
    process_batches(images, args)


if __name__ == "__main__":
    main()
