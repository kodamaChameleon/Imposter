# ⚙️ Usage

## Requirements

To utilize these scripts one must:
- Create a Mastodon account on mastodon.social and create an app token
- Create a BlueSky account and add an app password
- Create a Telegram bot and invite it into a group chat. Simplest method is using Telegrams [BotFather](https://telegram.me/BotFather)

## Installation

Copy the bots directory and the target dataset to transform to the location of your choosing.

*Optional* Use a virtual environment using:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required dependencies.
```bash
pip3 install -r requirements.txt
```

Copy the example environment and update environmental variables with your editor of choice.
```bash
cp .env_example .env
nano .env
```

## Execution

There are three standalone python scripts which all follow the same basic pattern:
- bskyBot.py
  - Max BATCH_SIZE: 4
  - Default DELAY: 120
- mastoBot.py
  - Max BATCH_SIZE: 4
  - Default DELAY: 240
- tgBot.py
  - Max BATCH_SIZE: 10
  - Default DELAY: 60

```
options:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
                        Path to a single image file
  --dir_path DIR_PATH   Path to a directory of images ( supports .png, .jpg,
                        .jpeg, .webp)
  --text TEXT           Text to accompany image posts
  --alt ALT             Alt text for images
  --out OUT             Directory to save downloaded images from post embeds
  --batch-size BATCH_SIZE
                        Number of images to post in one batch
  --delay DELAY         Seconds to wait between batches
  --delete              Whether to delete the post after downloading embed
                        images
```

**Example**
```bash
python3 tgBot.py --dir_path datasets/original/ --out datasets/telegram --delete --delay 90
```

> [!TIP]
> - Use a VPN or cloud hosted service for running these scripts.
> - Do **NOT** use personal accounts on BlueSky, Mastodon, or Telegram.
> - Respect the rate limit defaults. You can go slower if you want, but definitely don't go faster.
> - Use tmux to create simultaneous sessions which can be detached while running.
