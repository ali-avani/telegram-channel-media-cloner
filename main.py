import argparse
import asyncio
import glob
import os
import pickle
import time
from contextlib import suppress
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple, Union

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors.rpcerrorlist import FloodWaitError, MessageNotModifiedError
from telethon.tl.types import Message
from tqdm import tqdm

load_dotenv()


@dataclass
class Config:
    """Configuration class to hold all application settings."""

    api_id: str
    api_hash: str
    source_chat_id: int
    destination_channel_id: int
    base_dir: str
    media_path: str
    media_save_file_name: str
    show_chats: bool = False
    clean_channel: bool = False
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    ignore_database: bool = False
    dry_run: bool = False
    manual_pagination: bool = False

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from environment variables and command line arguments."""
        base_dir = os.path.dirname(os.path.realpath(__file__))
        return cls(
            api_id=os.environ.get("API_ID"),
            api_hash=os.environ.get("API_HASH"),
            source_chat_id=int(os.environ.get("SOURCE_CHAT_ID")),
            destination_channel_id=int(os.environ.get("DESTINATION_CHANNEL_ID")),
            base_dir=base_dir,
            media_path=f"{base_dir}/medias",
            media_save_file_name="saved_medias",
            show_chats=args.show_chats,
            clean_channel=args.clean_channel,
            start_id=args.start_id,
            end_id=args.end_id,
            ignore_database=args.ignore_database,
            dry_run=args.dry_run,
            manual_pagination=args.manual_pagination,
        )


class Timer:
    """Timer utility for rate limiting."""

    def __init__(self, time_between: float = 2):
        self.start_time = time.time()
        self.time_between = time_between

    def can_send(self) -> bool:
        """Check if enough time has passed since last send."""
        if time.time() > (self.start_time + self.time_between):
            self.start_time = time.time()
            return True
        return False


class MediaDB:
    """Database for tracking saved media files."""

    def __init__(self, filename: str):
        self.filename = filename
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as fp:
                self._saved_medias = set(pickle.load(fp))
        else:
            self._saved_medias = set()

    @property
    def saved_medias(self) -> set:
        return self._saved_medias

    def _save_to_file(self) -> None:
        """Save the current media set to file."""
        with open(self.filename, "wb") as fp:
            pickle.dump(self._saved_medias, fp)

    def add_media(self, media_id: Union[str, int]) -> None:
        """Add a media ID to the database."""
        self._saved_medias.add(media_id)
        self._save_to_file()

    def clean_media(self) -> None:
        """Clear all saved media from the database."""
        self._saved_medias = set()
        self._save_to_file()


class ProgressBar(tqdm):
    """Enhanced progress bar with Telegram message updates."""

    def __init__(self, client: TelegramClient, destination_channel_id: int, *args, **kwargs):
        self.client = client
        self.destination_channel_id = destination_channel_id
        self.start_fn = kwargs.pop("start_fn", None)
        self.progress_fn = kwargs.pop("progress_fn", None)
        super().__init__(*args, **kwargs)
        self.timer = Timer()
        self.message = None

    async def update_to(self, current: int, total: int) -> None:
        """Update progress and optionally send Telegram message updates."""
        self.total = total
        self.update(current - self.n)

        if self.start_fn and self.progress_fn:
            if not self.message:
                self.message = await self.client.send_message(self.destination_channel_id, self.start_fn(self))
            else:
                with suppress(FloodWaitError, MessageNotModifiedError):
                    if self.timer.can_send():
                        await self.message.edit(self.progress_fn(self))

            if current == total and self.message:
                await self.message.delete()


class FileProgressBar:
    """Progress bar formatter for file operations."""

    def __init__(self, message: Message, title: str, source_chat):
        self.message = message
        self.title = title
        self.source_chat = source_chat

    def get_message(self) -> str:
        """Get formatted message link."""
        return f"Message: https://t.me/c/{self.source_chat.id}/{self.message.id}"

    def progress(self, pbar: tqdm) -> str:
        """Format progress bar display."""
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

    def start_fn(self, _) -> str:
        """Get start message for progress."""
        return f"Starting {self.title}:\n{self.get_message()}"

    def progress_fn(self, pbar: tqdm) -> str:
        """Get progress message for updates."""
        return f"{self.get_message()}\n{self.progress(pbar)}"


class TelegramChannelMediaCloner:
    """Main application class for cloning media between Telegram channels."""

    def __init__(self, config: Config):
        self.config = config
        self.client = TelegramClient("tcmc_session", config.api_id, config.api_hash)
        self.media_db = MediaDB(config.media_save_file_name)
        self.source_chat = None
        self.destination_channel = None

        Path(config.media_path).mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the Telegram client."""
        await self.client.start()

    async def show_chats(self) -> None:
        """Display all available chats."""
        async for dialog in self.client.iter_dialogs():
            print(f"{dialog.name}: {dialog.id}")

    async def clean_channel(self) -> None:
        """Clean all messages from the destination channel."""
        message_ids = [
            message.id async for message in self.client.iter_messages(self.config.destination_channel_id, reverse=True)
        ]
        await self.client.delete_messages(
            entity=self.config.destination_channel_id,
            message_ids=message_ids,
        )
        self.media_db.clean_media()

    def _should_skip_message(self, message: Message) -> bool:
        """Check if a message should be skipped based on filters."""
        file = message.document or message.photo

        return (
            (self.config.start_id and message.id < self.config.start_id)
            or (self.config.end_id and message.id > self.config.end_id)
            or not file
            or (not self.config.ignore_database and file.id in self.media_db.saved_medias)
        )

    async def _fetch_messages_manual_pagination(self) -> List[Message]:
        """Fetch messages using manual pagination."""
        print("Using manual pagination to ensure all messages are retrieved...")
        messages = []
        message_count = 0
        offset_id = 0
        batch_size = 100

        while True:
            batch_messages = await self.client.get_messages(
                self.config.source_chat_id, limit=batch_size, offset_id=offset_id, reverse=True
            )

            if not batch_messages:
                break

            for message in batch_messages:
                message_count += 1
                if message_count % 1000 == 0:
                    print(f"Processed {message_count} messages...")

                if not self._should_skip_message(message):
                    messages.append(message)

            offset_id = batch_messages[-1].id
            await asyncio.sleep(0.1)

        return messages, message_count

    async def _fetch_messages_standard(self) -> Tuple[List[Message], int]:
        """Fetch messages using standard iteration."""
        messages = []
        message_count = 0

        async for message in self.client.iter_messages(
            self.config.source_chat_id,
            reverse=True,
            limit=None,
            wait_time=1,
        ):
            message_count += 1
            if message_count % 1000 == 0:
                print(f"Processed {message_count} messages...")

            if not self._should_skip_message(message):
                messages.append(message)

        return messages, message_count

    async def fetch_messages(self) -> List[Message]:
        """Fetch messages from the source chat."""
        source_name = self.source_chat.title if hasattr(self.source_chat, "title") else str(self.source_chat.id)
        print(f"Fetching messages from {source_name}...")

        if self.config.manual_pagination:
            messages, message_count = await self._fetch_messages_manual_pagination()
        else:
            messages, message_count = await self._fetch_messages_standard()

        print(f"Total messages processed: {message_count}")
        print(f"Messages with media found: {len(messages)}")

        return messages

    def display_dry_run_results(self, messages: List[Message]) -> None:
        """Display results for dry run mode."""
        print(f"Dry run - showing {len(messages)} messages with media:")
        for message in messages:
            message_link = f"https://t.me/c/{self.source_chat.id}/{message.id}"
            print(f"{message.id}: {message_link}")

    @staticmethod
    def display_upload_info(files: List[str]) -> None:
        """Display information about files being uploaded."""
        filenames = [file.split("/")[-1] for file in files]
        print(", ".join(filenames))

    async def _download_message_media(self, message: Message) -> Tuple[str, Optional[str], Optional[list]]:
        """Download media from a single message."""
        message = await self.client.get_messages(self.config.source_chat_id, ids=message.id)
        file = message.document or message.photo
        filename = file.id
        filepath_prefix = f"{self.config.media_path}/{filename}"
        is_video = not bool(message.photo)

        dpbr = FileProgressBar(message, "download", self.source_chat)
        with ProgressBar(
            self.client,
            self.config.destination_channel_id,
            unit="B",
            unit_scale=True,
            start_fn=dpbr.start_fn,
            progress_fn=dpbr.progress_fn,
        ) as progress_bar:
            file_path = await self.client.download_media(
                message, filepath_prefix, progress_callback=progress_bar.update_to
            )

        thumb_path = None
        attributes = None
        if is_video:
            thumb_path = await self.client.download_media(message, filepath_prefix, thumb=-1)
            attributes = file.attributes

        return file_path, thumb_path, attributes

    async def _upload_files(self, files: List[Tuple], message: Message) -> None:
        """Upload files to the destination channel."""
        print("Uploading:")

        if len(files) == 1:
            file_path, thumb_path, attributes = files[0]
            self.display_upload_info([file_path])
            files_to_send = file_path
            thumbs_to_send = thumb_path
            attributes_to_send = attributes
        elif len(files) > 1:
            file_paths, thumb_paths, attributes_list = zip(*files)
            self.display_upload_info(file_paths)
            files_to_send = file_paths
            thumbs_to_send = thumb_paths
            attributes_to_send = attributes_list
        else:
            return

        send_file_args = {
            "entity": self.destination_channel,
            "file": files_to_send,
            "thumb": thumbs_to_send,
            "supports_streaming": True,
            "attributes": attributes_to_send,
        }

        upbr = FileProgressBar(message, "upload", self.source_chat)
        with ProgressBar(
            self.client,
            self.config.destination_channel_id,
            unit="B",
            unit_scale=True,
            start_fn=upbr.start_fn,
            progress_fn=upbr.progress_fn,
        ) as progress_bar:
            send_file_args["progress_callback"] = progress_bar.update_to
            await self.client.send_file(**send_file_args)

    def _cleanup_files(self, filenames: List[Union[str, int]]) -> None:
        """Clean up downloaded files."""
        print("Cleaning up")
        for filename in filenames:
            for file_path in glob.glob(f"{self.config.media_path}/{filename}*"):
                with suppress(OSError):
                    os.remove(file_path)

    async def process_messages(self, messages: List[Message]) -> None:
        """Process and clone messages with media."""
        for key, group in groupby(messages, lambda x: x.grouped_id or x.id):
            files = []
            filenames = []

            try:
                print("Downloading:", key)

                for message in group:
                    file_path, thumb_path, attributes = await self._download_message_media(message)
                    files.append((file_path, thumb_path, attributes))

                    file = message.document or message.photo
                    filenames.append(file.id)

                await self._upload_files(files, message)

                for filename in filenames:
                    self.media_db.add_media(filename)

            finally:
                self._cleanup_files(filenames)
                print("Done")
                print()

    async def run(self) -> None:
        """Main execution method."""
        await self.start()

        if self.config.show_chats:
            await self.show_chats()
            return

        if self.config.clean_channel:
            await self.clean_channel()
            return

        self.source_chat = await self.client.get_entity(self.config.source_chat_id)
        self.destination_channel = await self.client.get_entity(self.config.destination_channel_id)

        messages = await self.fetch_messages()

        if self.config.dry_run:
            self.display_dry_run_results(messages)
            return

        await self.process_messages(messages)


def check_positive(value: str) -> int:
    """Validate that a string represents a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


def init_argparse() -> argparse.ArgumentParser:
    """Initialize and configure argument parser."""
    parser = argparse.ArgumentParser(description="Clone media from one Telegram channel to another")
    parser.add_argument("-p", "--show-chats", action="store_true", help="Show all chats")
    parser.add_argument("-c", "--clean-channel", action="store_true", help="Clean channel all messages")
    parser.add_argument("-s", "--start-id", type=check_positive, help="Start from message id")
    parser.add_argument("-e", "--end-id", type=check_positive, help="End at message id")
    parser.add_argument("-d", "--ignore-database", action="store_true", help="Ignore media database")
    parser.add_argument("--dry-run", action="store_true", help="Dry run and only show messages")
    parser.add_argument(
        "--manual-pagination", action="store_true", help="Use manual pagination to ensure all messages are retrieved"
    )
    return parser


async def main() -> None:
    """Main entry point."""
    parser = init_argparse()
    args = parser.parse_args()

    config = Config.from_env_and_args(args)
    cloner = TelegramChannelMediaCloner(config)

    await cloner.run()


if __name__ == "__main__":
    asyncio.run(main())
