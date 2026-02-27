"""
-*- coding: utf-8 -*-
Name:        bskyBot.py
Purpose:     Filter images through a BlueSky upload/post cycle
Author:      Kodama Chameleon <contact@kodamachameleon.com>
"""
import argparse
import os
import time
from pathlib import Path
from typing import List

import requests
from atproto import Client
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
MAX_BATCH = 4
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# ---------- helpers ----------

def aspect_ratio_dict(path: str):
    try:
        with Image.open(path) as im:
            w, h = im.size
        return {"width": int(w), "height": int(h)}
    except Exception:
        return None


def extract_rkey(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def create_client(args) -> Client:
    if not args.handle or not args.app_password:
        raise SystemExit("Missing credentials")

    client = Client(base_url=args.pds)
    client.login(args.handle, args.app_password)
    return client


# ---------- inputs ----------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", help="Path to a single image file")
    parser.add_argument("--dir_path", help="Path to a directory of images ( supports .png, .jpg, .jpeg, .webp)")

    parser.add_argument("--text", default="Posted from Python 🐍", help="Text to accompany image posts")
    parser.add_argument("--alt", default="Image posted from Python", help="Alt text for images")

    parser.add_argument("--out", default="datasets/results", help="Directory to save downloaded images from post embeds")

    parser.add_argument("--pds", default=os.getenv("BSKY_PDS", "https://bsky.social"), help="Base URL of the PDS (defaults to bsky.social or env BSKY_PDS)")
    parser.add_argument("--handle", default=os.getenv("BSKY_HANDLE"), help="Your Bsky handle (defaults to env BSKY_HANDLE)")
    parser.add_argument("--app-password", default=os.getenv("BSKY_APP_PASSWORD"), help="Your Bsky app password (defaults to env BSKY_APP_PASSWORD)")

    parser.add_argument("--batch-size", type=int, default=4, help=f"Number of images to post in one batch (max {MAX_BATCH})")
    parser.add_argument("--delay", type=int, default=300, help="Seconds to wait between batches")

    parser.add_argument("--delete", action="store_true", help="Whether to delete the post after downloading embed images")

    args = parser.parse_args()

    if not args.image_path and not args.dir_path:
        raise SystemExit("Provide --image_path or --dir_path")

    if args.image_path and args.dir_path:
        raise SystemExit("Provide only one of --image_path or --dir_path")

    args.batch_size = min(args.batch_size, MAX_BATCH)

    os.makedirs(args.out, exist_ok=True)

    return args


def collect_images(args) -> List[Path]:
    if args.image_path:
        return [Path(args.image_path)]

    files = [
        p for p in Path(args.dir_path).iterdir()
        if p.suffix.lower() in IMG_EXTS
    ]

    return sorted(files)


# ---------- outputs ----------

def download_images(embed_images, src_paths, out_dir):
    for img_meta, src in zip(embed_images, src_paths):

        stem = src.stem
        ext = src.suffix

        full_url = img_meta["fullsize"]
        thumb_url = img_meta["thumb"]

        full_path = Path(out_dir) / f"{stem}_full{ext}"
        thumb_path = Path(out_dir) / f"{stem}_thumb{ext}"

        for url, path in [(full_url, full_path), (thumb_url, thumb_path)]:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

        print(f"Downloaded: {stem}")


def process_batches(client: Client, images: List[Path], args):

    for i in range(0, len(images), args.batch_size):

        batch = images[i:i + args.batch_size]

        img_bytes = [p.read_bytes() for p in batch]
        ars = [aspect_ratio_dict(p) for p in batch]

        kwargs = dict(
            text=args.text,
            images=img_bytes,
            image_alts=[args.alt] * len(batch),
        )

        if all(ars):
            kwargs["image_aspect_ratios"] = ars

        post = client.send_images(**kwargs)
        post_uri = post.uri

        print(f"Posted: {post_uri}")

        success = False

        for attempt in range(5):
            try:
                time.sleep(1)

                thread = client.app.bsky.feed.get_post_thread(
                    {"uri": post_uri}
                ).model_dump()

                post = thread["thread"]["post"]
                embed = post.get("embed")

                if not embed or "images" not in embed:
                    raise ValueError("Embed not hydrated yet")

                download_images(embed["images"], batch, args.out)

                success = True
                break

            except Exception as e:
                print(f"[attempt {attempt+1}/5] waiting for embed: {e}")

        if not success:
            raise RuntimeError(f"Failed to download images for post {post_uri}")

        if args.delete:
            rkey = extract_rkey(post_uri)
            client.com.atproto.repo.delete_record({
                "repo": client.me.did,
                "collection": "app.bsky.feed.post",
                "rkey": rkey,
            })
            print("Deleted post")

        if i + args.batch_size < len(images):
            print(f"Sleeping {args.delay}s...")
            time.sleep(args.delay)


# ---------- main ----------

def main():
    args = parse_args()
    images = collect_images(args)
    client = create_client(args)
    process_batches(client, images, args)


if __name__ == "__main__":
    main()
