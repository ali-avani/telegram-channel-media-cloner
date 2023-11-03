import asyncio
import glob
import os
import pickle
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from tqdm import tqdm

load_dotenv()

API_ID = os.environ.get("API_ID")
API_HASH = os.environ.get("API_HASH")
SOURCE_CHAT_ID = int(os.environ.get("SOURCE_CHAT_ID"))
DESTINATION_CHANNEL_ID = int(os.environ.get("DESTINATION_CHANNEL_ID"))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MEDIA_PATH = f"{BASE_DIR}/medias"
MEDIA_SAVE_FILE_NAME = "saved_medias"

loop = asyncio.get_event_loop()
client = TelegramClient("tcmc_session", API_ID, API_HASH)

Path(MEDIA_PATH).mkdir(parents=True, exist_ok=True)


class MediaDB:
    filename = None
    _saved_medias = []

    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as fp:
                self._saved_medias = set(pickle.load(fp))
        else:
            self._saved_medias = set()

    @property
    def saved_medias(self):
        return self._saved_medias

    @saved_medias.setter
    def saved_medias(self, value):
        with open(self.filename, "wb") as fp:
            pickle.dump(value, fp)
        return self._saved_medias

    def add_media(self, media):
        self._saved_medias.add(media)
        self.saved_medias = self._saved_medias


class DownloadProgressBar(tqdm):
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)


async def main():
    media_db = MediaDB(MEDIA_SAVE_FILE_NAME)
    await client.start()
    channel = await client.get_entity(DESTINATION_CHANNEL_ID)
    messages = []
    async for message in client.iter_messages(SOURCE_CHAT_ID, reverse=True):
        file = message.document or message.photo
        if file:
            if not file.id in media_db.saved_medias:
                messages.append(message)
    for key, group in groupby(messages, lambda x: x.grouped_id or x.id):
        try:
            print("Downloading:", key)
            files = []
            filenames = []
            for message in group:
                file = message.document or message.photo
                filename = file.id
                filepath_prefix = f"{MEDIA_PATH}/{filename}"
                is_video = not bool(message.photo)
                if not filename in media_db.saved_medias:
                    with DownloadProgressBar(unit="B", unit_scale=True) as t:
                        file_path = await client.download_media(message, filepath_prefix, progress_callback=t.update_to)
                    thumb_path = None
                    if is_video:
                        thumb_path = await client.download_media(message, filepath_prefix, thumb=-1)
                        send_file_args.update({"thumb": thumb_path, "supports_streaming": True})
                    files.append((file_path, thumb_path))
                    filenames.append(filename)
            print("Uploading:")
            send_file_args = {
                "entity": channel,
                "file": [f for f, _ in files],
                "thumb": [t for _, t in files],
                "supports_streaming": True,
            }
            with DownloadProgressBar(unit="B", unit_scale=True) as t:
                send_file_args.update({"progress_callback": t.update_to})
                await client.send_file(**send_file_args)
            try:
                for filename in filenames:
                    for f in glob.glob(f"{MEDIA_PATH}/{filename}" + "*"):
                        os.remove(f)
            except OSError:
                pass
        finally:
            print("Done")
            for filename in filenames:
                media_db.add_media(filename)


if __name__ == "__main__":
    loop.run_until_complete(main())
