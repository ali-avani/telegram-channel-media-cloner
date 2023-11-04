import argparse
import asyncio
import glob
import os
import pickle
import time
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

show_chats = False
clean_channel = False

loop = asyncio.get_event_loop()
client = TelegramClient("tcmc_session", API_ID, API_HASH)

Path(MEDIA_PATH).mkdir(parents=True, exist_ok=True)


class Timer:
    def __init__(self, time_between=1):
        self.start_time = time.time()
        self.time_between = time_between

    def can_send(self):
        if time.time() > (self.start_time + self.time_between):
            self.start_time = time.time()
            return True
        return False


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

    def clean_media(self):
        self.saved_medias = set()


class ProgressBar(tqdm):
    timer = None
    message = None
    start_fn = None
    progress_fn = None

    def __init__(self, *args, **kwargs):
        self.start_fn = kwargs.pop("start_fn", None)
        self.progress_fn = kwargs.pop("progress_fn", None)
        super().__init__(*args, **kwargs)
        self.timer = Timer()

    async def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)
        if self.start_fn and self.progress_fn:
            if not self.message:
                self.message = await client.send_message(DESTINATION_CHANNEL_ID, self.start_fn(self))
            else:
                if self.timer.can_send():
                    await self.message.edit(self.progress_fn and self.progress_fn(self))
            if current == total:
                await self.message.delete()


class DownloadProgressBar:
    message = None

    def __init__(self, message):
        self.message = message

    def get_message(self):
        return f"Message: {self.message.id}"

    def download_progress(self, progress_bar):
        return f"Downloading: {tqdm.format_sizeof(progress_bar.n)}/{tqdm.format_sizeof(progress_bar.total)} "

    def download_start_fn(self, progress_bar):
        return f"Starting:\n{self.get_message()}"

    def download_progress_fn(self, progress_bar):
        return f"{self.get_message()}\n{self.download_progress(progress_bar)}"


class UploadProgressBar:
    message = None

    def __init__(self, message):
        self.message = message

    def get_message(self):
        return f"Message: {self.message.id}"

    def upload_progress(self, progress_bar):
        return f"Uploading: {tqdm.format_sizeof(progress_bar.n)}/{tqdm.format_sizeof(progress_bar.total)} "

    def upload_start_fn(self, progress_bar):
        return f"Starting:\n{self.get_message()}"

    def upload_progress_fn(self, progress_bar):
        return f"{self.get_message()}\n{self.upload_progress(progress_bar)}"


def display_upload_info(files):
    print(", ".join([file.split("/")[-1] for file in files]))


async def main():
    media_db = MediaDB(MEDIA_SAVE_FILE_NAME)
    await client.start()

    if show_chats:
        async for dialog in client.iter_dialogs():
            print(f"{dialog.name}: {dialog.id}")
        return

    if clean_channel:
        await client.delete_messages(
            entity=DESTINATION_CHANNEL_ID,
            message_ids=[message.id async for message in client.iter_messages(DESTINATION_CHANNEL_ID, reverse=True)],
        )
        media_db.clean_media()
        return

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
                message = await client.get_messages(SOURCE_CHAT_ID, ids=message.id)
                file = message.document or message.photo
                filename = file.id
                filepath_prefix = f"{MEDIA_PATH}/{filename}"
                is_video = not bool(message.photo)
                if not filename in media_db.saved_medias:
                    thumb_path = None
                    attributes = None
                    dpbr = DownloadProgressBar(message)
                    with ProgressBar(
                        unit="B",
                        unit_scale=True,
                        start_fn=dpbr.download_start_fn,
                        progress_fn=dpbr.download_progress_fn,
                    ) as t:
                        file_path = await client.download_media(message, filepath_prefix, progress_callback=t.update_to)
                    if is_video:
                        thumb_path = await client.download_media(message, filepath_prefix, thumb=-1)
                        attributes = file.attributes
                    files.append((file_path, thumb_path, attributes))
                    filenames.append(filename)
            print("Uploading:")
            if len(files) == 1:
                files, thumbs, attributes = files[0]
                display_upload_info([files])
            elif len(files) > 1:
                files, thumbs, attributes = zip(*files)
                display_upload_info(files)
            else:
                continue
            send_file_args = {
                "entity": channel,
                "file": files,
                "thumb": thumbs,
                "supports_streaming": True,
                "attributes": attributes,
            }
            upbr = UploadProgressBar(message)
            with ProgressBar(
                unit="B",
                unit_scale=True,
                start_fn=upbr.upload_start_fn,
                progress_fn=upbr.upload_progress_fn,
            ) as t:
                send_file_args.update({"progress_callback": t.update_to})
                await client.send_file(**send_file_args)
        finally:
            print("Cleaning up:")
            try:
                for filename in filenames:
                    for f in glob.glob(f"{MEDIA_PATH}/{filename}" + "*"):
                        os.remove(f)
            except OSError:
                pass
            for filename in filenames:
                media_db.add_media(filename)

            print("Done")
            print()


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action="store_true")
    parser.add_argument("-c", action="store_true")
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    if args.s:
        show_chats = True
    if args.c:
        clean_channel = True

    loop.run_until_complete(main())
