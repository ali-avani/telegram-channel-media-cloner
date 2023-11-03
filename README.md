# Shekan Installer

Python script to download all of media from private Telegram channel and upload them to custom channel.

## Getting Started

Go to [link](https://my.telegram.org/) and create an app. Then get your `api_id` and `api_hash`.

## Installation

Download the scripts:

```
git clone https://github.com/ali-avani/telegram-channel-media-cloner.git
cd telegram-channel-media-cloner
```

Edit configuration:

```
cp .env.sample .env
vim .env
```

Run the installer:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```
