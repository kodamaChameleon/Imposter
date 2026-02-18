"""
-*- coding: utf-8 -*-
Name:        tgBot.py
Purpose:     Filter images through a Telegram upload/post cycle
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
MAX_BATCH = 10
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------- helpers ----------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path")
    parser.add_argument("--dir_path")

    parser.add_argument("--text", default="Posted from Python 🐍")
    parser.add_argument("--out", default="datasets/results")

    parser.add_argument("--bot-token", default=os.getenv("TG_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.getenv("TG_CHAT_ID"))

    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay", type=int, default=10)

    parser.add_argument("--delete", action="store_true")

    args = parser.parse_args()

    if not args.image_path and not args.dir_path:
        raise SystemExit("Provide --image_path or --dir_path")

    if args.image_path and args.dir_path:
        raise SystemExit("Only one input source allowed")

    if not args.bot_token or not args.chat_id:
        raise SystemExit("Missing Telegram credentials")

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


# ---------- telegram core ----------

def send_batch(token, chat_id, batch, caption):
    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"

    media = []
    files = {}

    for i, path in enumerate(batch):
        key = f"file{i}"
        files[key] = path.read_bytes()

        media.append({
            "type": "photo",
            "media": f"attach://{key}",
            "caption": caption if i == 0 else ""
        })

    r = requests.post(
        url,
        data={"chat_id": chat_id, "media": str(media).replace("'", '"')},
        files=files,
        timeout=60
    )
    r.raise_for_status()

    return r.json()["result"]


def get_file_url(token, file_id):
    meta = requests.get(
        f"https://api.telegram.org/bot{token}/getFile",
        params={"file_id": file_id},
        timeout=30
    ).json()

    path = meta["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{token}/{path}"


def download_images(token, messages, batch, out_dir):
    for msg, src in zip(messages, batch):

        sizes = msg["photo"]  # smallest → largest

        stem = src.stem
        ext = src.suffix

        for size in sizes:
            file_id = size["file_id"]
            w = size["width"]
            h = size["height"]

            url = get_file_url(token, file_id)

            tag = f"{w}x{h}"

            out_path = Path(out_dir) / f"{stem}_{tag}{ext}"

            r = requests.get(url, timeout=60)
            r.raise_for_status()

            with open(out_path, "wb") as f:
                f.write(r.content)

        print(f"Downloaded all sizes for: {stem}")


def delete_messages(token, chat_id, messages):
    for msg in messages:
        requests.post(
            f"https://api.telegram.org/bot{token}/deleteMessage",
            data={
                "chat_id": chat_id,
                "message_id": msg["message_id"]
            }
        )


# ---------- batching ----------

def process_batches(images, args):

    for i in range(0, len(images), args.batch_size):

        batch = images[i:i + args.batch_size]

        messages = send_batch(
            args.bot_token,
            args.chat_id,
            batch,
            args.text
        )

        print("Posted batch")

        download_images(args.bot_token, messages, batch, args.out)

        if args.delete:
            delete_messages(args.bot_token, args.chat_id, messages)
            print("Deleted batch")

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
