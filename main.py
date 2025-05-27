import argparse
import asyncio
import glob
import logging
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


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"


# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Colors.BLUE + "%(asctime)s - %(levelname)s - %(message)s" + Colors.RESET,
        logging.INFO: Colors.GREEN + "%(asctime)s - %(levelname)s - %(message)s" + Colors.RESET,
        logging.WARNING: Colors.YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + Colors.RESET,
        logging.ERROR: Colors.RED + "%(asctime)s - %(levelname)s - %(message)s" + Colors.RESET,
        logging.CRITICAL: Colors.RED + Colors.BOLD + "%(asctime)s - %(levelname)s - %(message)s" + Colors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())

# Set root logger level to CRITICAL to suppress most library logs
logging.getLogger().setLevel(logging.CRITICAL)

# Create our app logger with appropriate level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# Silence Telethon logs specifically
logging.getLogger("telethon").setLevel(logging.CRITICAL)


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
        logger.info("Retrieving available chats...")
        chat_count = 0
        async for dialog in self.client.iter_dialogs():
            chat_count += 1
            logger.info(f"  {dialog.name}: {dialog.id}")
        logger.info(f"Found {chat_count} chats total")

    async def clean_channel(self) -> None:
        """Clean all messages from the destination channel."""
        logger.info("Starting channel cleanup...")
        logger.info("Retrieving messages to delete...")

        message_ids = [
            message.id async for message in self.client.iter_messages(self.config.destination_channel_id, reverse=True)
        ]

        if not message_ids:
            logger.info("Channel is already empty")
            return

        logger.info(f"Deleting {len(message_ids)} messages from destination channel...")
        await self.client.delete_messages(
            entity=self.config.destination_channel_id,
            message_ids=message_ids,
        )

        logger.info("Clearing media database...")
        self.media_db.clean_media()
        logger.info("Channel cleanup completed successfully")

    def _should_skip_message(self, message: Message) -> bool:
        """Check if a message should be skipped based on filters."""
        file = message.document or message.photo

        return (
            (self.config.start_id and message.id < self.config.start_id)
            or (self.config.end_id and message.id > self.config.end_id)
            or not file
            or (not self.config.ignore_database and file.id in self.media_db.saved_medias)
        )

    async def _fetch_all_messages(self) -> Tuple[List[Message], int]:
        """Fetch all messages from the source chat using pagination."""
        logger.info("Fetching messages using pagination for complete retrieval...")
        messages = []
        message_count = 0
        offset_id = 0
        batch_size = 100
        batches_processed = 0

        while True:
            batch_messages = await self.client.get_messages(
                self.config.source_chat_id, limit=batch_size, offset_id=offset_id, reverse=True
            )

            if not batch_messages:
                break

            batches_processed += 1
            for message in batch_messages:
                message_count += 1
                if message_count % 1000 == 0:
                    logger.info(f"Processed {message_count:,} messages so far...")

                if not self._should_skip_message(message):
                    messages.append(message)

            offset_id = batch_messages[-1].id
            await asyncio.sleep(0.1)

        logger.info(f"Processed {batches_processed} batches with {batch_size} messages each")
        return messages, message_count

    async def fetch_messages(self) -> List[Message]:
        """Fetch messages from the source chat."""
        source_name = self.source_chat.title if hasattr(self.source_chat, "title") else str(self.source_chat.id)
        logger.info(f"Starting message fetch from: {source_name}")

        messages, message_count = await self._fetch_all_messages()

        logger.info(f"Total messages processed: {message_count:,}")
        logger.info(f"Messages with media found: {len(messages):,}")

        if len(messages) == 0:
            logger.warning("No media messages found to process")
        else:
            logger.info(f"Ready to process {len(messages):,} media messages")

        return messages

    def display_dry_run_results(self, messages: List[Message]) -> None:
        """Display results for dry run mode."""
        logger.info(f"DRY RUN MODE - Found {len(messages):,} messages with media:")
        logger.info("Message list:")

        for i, message in enumerate(messages, 1):
            message_link = f"https://t.me/c/{self.source_chat.id}/{message.id}"
            file_type = "Video" if message.document else "Photo"
            logger.info(f"  {i:3d}. {file_type} - ID: {message.id} - Link: {message_link}")

        logger.info(f"Dry run complete - {len(messages):,} media files would be processed")

    @staticmethod
    def display_upload_info(files: List[str]) -> None:
        """Display information about files being uploaded."""
        filenames = [file.split("/")[-1] for file in files]
        if len(filenames) == 1:
            logger.info(f"Uploading file: {filenames[0]}")
        else:
            logger.info(f"Uploading {len(filenames)} files: {', '.join(filenames)}")

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
        logger.debug("Cleaning up temporary files...")
        files_removed = 0
        for filename in filenames:
            for file_path in glob.glob(f"{self.config.media_path}/{filename}*"):
                with suppress(OSError):
                    os.remove(file_path)
                    files_removed += 1
        if files_removed > 0:
            logger.debug(f"Removed {files_removed} temporary files")

    async def process_messages(self, messages: List[Message]) -> None:
        """Process and clone messages with media."""
        total_groups = len(list(groupby(messages, lambda x: x.grouped_id or x.id)))
        processed_groups = 0

        logger.info(f"Starting media processing for {total_groups} message groups...")

        for key, group in groupby(messages, lambda x: x.grouped_id or x.id):
            files = []
            filenames = []
            processed_groups += 1

            try:
                group_list = list(group)
                logger.info(f"[{processed_groups}/{total_groups}] Processing group {key} ({len(group_list)} files)")

                for i, message in enumerate(group_list, 1):
                    logger.info(f"  Downloading file {i}/{len(group_list)}...")
                    file_path, thumb_path, attributes = await self._download_message_media(message)
                    files.append((file_path, thumb_path, attributes))

                    file = message.document or message.photo
                    filenames.append(file.id)

                await self._upload_files(files, message)

                logger.info("Updating media database...")
                for filename in filenames:
                    self.media_db.add_media(filename)

                logger.info(f"Group {key} completed successfully")

            except Exception as e:
                logger.error(f"Error processing group {key}: {str(e)}")
                raise
            finally:
                self._cleanup_files(filenames)

        logger.info(f"All done! Successfully processed {processed_groups} message groups")

    async def run(self) -> None:
        """Main execution method."""
        logger.info("Starting Telegram Channel Media Cloner...")

        try:
            logger.info("Connecting to Telegram...")
            await self.start()
            logger.info("Successfully connected to Telegram")

            if self.config.show_chats:
                await self.show_chats()
                return

            if self.config.clean_channel:
                await self.clean_channel()
                return

            logger.info("Retrieving chat entities...")
            self.source_chat = await self.client.get_entity(self.config.source_chat_id)
            self.destination_channel = await self.client.get_entity(self.config.destination_channel_id)

            source_name = self.source_chat.title if hasattr(self.source_chat, "title") else str(self.source_chat.id)
            dest_name = (
                self.destination_channel.title
                if hasattr(self.destination_channel, "title")
                else str(self.destination_channel.id)
            )

            logger.info(f"Source: {source_name}")
            logger.info(f"Destination: {dest_name}")

            messages = await self.fetch_messages()

            if self.config.dry_run:
                self.display_dry_run_results(messages)
                return

            if len(messages) == 0:
                logger.info("No media messages to process. Exiting.")
                return

            await self.process_messages(messages)
            logger.info("Media cloning completed successfully!")

        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            raise


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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (debug level)")
    return parser


async def main() -> None:
    """Main entry point."""
    parser = init_argparse()
    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    config = Config.from_env_and_args(args)
    cloner = TelegramChannelMediaCloner(config)

    await cloner.run()


if __name__ == "__main__":
    asyncio.run(main())
