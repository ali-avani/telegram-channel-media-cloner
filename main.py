import argparse
import asyncio
import glob
import os
import pickle
import time
from contextlib import suppress
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors.rpcerrorlist import FloodWaitError, MessageNotModifiedError
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
source_chat = None


show_chats = False
clean_channel = False
start_id = None
end_id = None
ignore_database = False
dry_run = False
use_takeout = False
manual_pagination = False

Path(MEDIA_PATH).mkdir(parents=True, exist_ok=True)


class Timer:
    def __init__(self, time_between=2):
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
                with suppress(FloodWaitError, MessageNotModifiedError):
                    if self.timer.can_send():
                        await self.message.edit(self.progress_fn(self))
            if current == total:
                await self.message.delete()


class FileProgressBar:
    message = None
    title = None

    def __init__(self, message, title):
        self.message = message
        self.title = title

    def get_message(self):
        return f"Message: {get_message_link(self.message)}"

    def progress(self, pbar):
        title = f"{self.title.capitalize()}ing"
        return pbar.format_meter(
            pbar.n,
            pbar.total,
            pbar.format_dict["elapsed"],
            prefix=title,
            ncols=0,
            unit="B",
            unit_scale=True,
        )

    def start_fn(self, _):
        return f"Starting {self.title}:\n{self.get_message()}"

    def progress_fn(self, pbar):
        return f"{self.get_message()}\n{self.progress(pbar)}"


def display_upload_info(files):
    print(", ".join([file.split("/")[-1] for file in files]))


def get_message_link(message):
    return f"https://t.me/c/{source_chat.id}/{message.id}"


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


async def main():
    global source_chat
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

    source_chat = await client.get_entity(SOURCE_CHAT_ID)
    destination_channel = await client.get_entity(DESTINATION_CHANNEL_ID)
    messages = []

    print(f"Fetching messages from {source_chat.title if hasattr(source_chat, 'title') else source_chat.id}...")
    message_count = 0

    takeout_success = False
    if use_takeout:
        print("Using takeout session for bulk export...")
        try:
            async with client.takeout(megagroups=True, channels=True) as takeout:
                async for message in takeout.iter_messages(
                    SOURCE_CHAT_ID,
                    reverse=True,
                    limit=None,
                    wait_time=0,
                ):
                    message_count += 1
                    if message_count % 1000 == 0:
                        print(f"Processed {message_count} messages...")

                    file = message.document or message.photo
                    if (
                        (start_id and message.id < start_id)
                        or (end_id and message.id > end_id)
                        or not file
                        or (not ignore_database and file.id in media_db.saved_medias)
                    ):
                        continue
                    messages.append(message)
                takeout_success = True
        except Exception as e:
            print(f"Takeout failed: {e}")
            print("Falling back to regular method...")

    if not takeout_success:
        if manual_pagination:
            print("Using manual pagination to ensure all messages are retrieved...")
            offset_id = 0
            batch_size = 100

            while True:
                batch_messages = await client.get_messages(
                    SOURCE_CHAT_ID, limit=batch_size, offset_id=offset_id, reverse=True
                )

                if not batch_messages:
                    break

                for message in batch_messages:
                    message_count += 1
                    if message_count % 1000 == 0:
                        print(f"Processed {message_count} messages...")

                    file = message.document or message.photo
                    if (
                        (start_id and message.id < start_id)
                        or (end_id and message.id > end_id)
                        or not file
                        or (not ignore_database and file.id in media_db.saved_medias)
                    ):
                        continue
                    messages.append(message)

                offset_id = batch_messages[-1].id

                await asyncio.sleep(0.1)
        else:
            async for message in client.iter_messages(
                SOURCE_CHAT_ID,
                reverse=True,
                limit=None,
                wait_time=1,
            ):
                message_count += 1
                if message_count % 1000 == 0:
                    print(f"Processed {message_count} messages...")

                file = message.document or message.photo
                if (
                    (start_id and message.id < start_id)
                    or (end_id and message.id > end_id)
                    or not file
                    or (not ignore_database and file.id in media_db.saved_medias)
                ):
                    continue
                messages.append(message)

    print(f"Total messages processed: {message_count}")
    print(f"Messages with media found: {len(messages)}")

    if dry_run:
        print(f"Dry run - showing {len(messages)} messages with media:")
        for message in messages:
            print(f"{message.id}: {get_message_link(message)}")
        return

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
                thumb_path = None
                attributes = None
                dpbr = FileProgressBar(message, "download")
                with ProgressBar(
                    unit="B",
                    unit_scale=True,
                    start_fn=dpbr.start_fn,
                    progress_fn=dpbr.progress_fn,
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
                "entity": destination_channel,
                "file": files,
                "thumb": thumbs,
                "supports_streaming": True,
                "attributes": attributes,
            }
            upbr = FileProgressBar(message, "upload")
            with ProgressBar(
                unit="B",
                unit_scale=True,
                start_fn=upbr.start_fn,
                progress_fn=upbr.progress_fn,
            ) as t:
                send_file_args.update({"progress_callback": t.update_to})
                await client.send_file(**send_file_args)
            for filename in filenames:
                media_db.add_media(filename)
        finally:
            print("Cleaning up")
            for filename in filenames:
                for f in glob.glob(f"{MEDIA_PATH}/{filename}" + "*"):
                    with suppress(OSError):
                        os.remove(f)
            print("Done")
            print()


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--show-chats", action="store_true", help="Show all chats")
    parser.add_argument("-c", "--clean-channel", action="store_true", help="Clean channel all messages")
    parser.add_argument("-s", "--start-id", type=check_positive, help="Start from message id")
    parser.add_argument("-e", "--end-id", type=check_positive, help="End at message id")
    parser.add_argument("-d", "--ignore-database", action="store_true", help="Ignore media database")
    parser.add_argument("--dry-run", action="store_true", help="Dry run and only show messages")
    parser.add_argument(
        "--use-takeout", action="store_true", help="Use takeout session for bulk export (lower rate limits)"
    )
    parser.add_argument(
        "--manual-pagination", action="store_true", help="Use manual pagination to ensure all messages are retrieved"
    )
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    if args.show_chats:
        show_chats = True
    if args.clean_channel:
        clean_channel = True
    if args.start_id:
        start_id = args.start_id
    if args.end_id:
        end_id = args.end_id
    if args.ignore_database:
        ignore_database = True
    if args.dry_run:
        dry_run = True
    if args.use_takeout:
        use_takeout = True
    if args.manual_pagination:
        manual_pagination = True

    loop.run_until_complete(main())
