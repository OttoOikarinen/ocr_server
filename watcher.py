#!/usr/bin/env python3
"""
watcher.py – Watchdog file-system monitor for the Finnish OCR pipeline.

Watches ~/ocr_pipeline/incoming/ recursively for new image files.
Each immediate subdirectory maps to a book name:

    ~/ocr_pipeline/incoming/{kirjan_nimi}/{sivu}.tif

When a new image lands, it is queued for OCR via ocr_engine.process_book_image().

Usage:
    python watcher.py
    python watcher.py --workers 4
    python watcher.py --no-backlog --debug
"""

import argparse
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ocr_engine import process_book_image


# ---------------------------------------------------------------------------
# Logging – stderr only, same format as ocr_engine.py
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSING_DIR = Path.home() / "ocr_pipeline" / "incoming"

# All image formats that OpenCV (and therefore ocr_engine) can open
SUPPORTED_EXTENSIONS = frozenset({
    ".tif", ".tiff",
    ".jpg", ".jpeg",
    ".png",
    ".bmp",
    ".webp",
})

# Polling interval while waiting for a file write to complete (seconds)
STABILITY_POLL_INTERVAL = 0.5

# Give up waiting for a file to stabilise after this many seconds.
# Large TIFF files from dedicated book scanners can be 50–200 MB.
STABILITY_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# File-stability helper
# ---------------------------------------------------------------------------

def _wait_for_stable(filepath: Path) -> bool:
    """
    Poll the file size until it is unchanged for two consecutive readings.

    Returns True when the file is stable and non-empty, False on timeout.

    Why this is necessary: Watchdog's on_created fires the moment the OS
    creates the file entry, which is well before a scanner's software or a
    network copy has finished writing the data.  Starting OCR on a partial
    file would produce garbage text and corrupt the master output.
    """
    deadline = time.monotonic() + STABILITY_TIMEOUT
    prev_size = -1

    while time.monotonic() < deadline:
        try:
            current_size = filepath.stat().st_size
        except FileNotFoundError:
            logger.warning("File disappeared while waiting for stability: %s", filepath)
            return False

        if current_size > 0 and current_size == prev_size:
            return True  # Unchanged for a full interval → write complete

        prev_size = current_size
        time.sleep(STABILITY_POLL_INTERVAL)

    logger.warning(
        "Timed out after %.0f s waiting for '%s' to stabilise.", STABILITY_TIMEOUT, filepath.name
    )
    return False


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _is_image(filepath: Path) -> bool:
    return filepath.suffix.lower() in SUPPORTED_EXTENSIONS


def _book_name(filepath: Path) -> str | None:
    """
    Derive the book name from the file's immediate parent directory.

    Expected layout: PROCESSING_DIR / {book_name} / {scan_file}
    Returns None for files not following this two-level structure.
    """
    try:
        rel = filepath.relative_to(PROCESSING_DIR)
    except ValueError:
        return None
    # rel.parts must be exactly ('kirjan_nimi', 'sivu001.tif')
    return rel.parts[0] if len(rel.parts) == 2 else None


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------

class OCREventHandler(FileSystemEventHandler):
    """
    Queues new image files for OCR processing via a thread pool.

    Thread safety: on_created / on_moved are called from the single Watchdog
    observer thread, but _run() executes in the pool.  The _in_flight set is
    guarded by a lock to prevent double-queuing the same file (e.g. when both
    an IN_CREATE and an IN_MODIFY event fire for the same write).
    """

    def __init__(self, executor: ThreadPoolExecutor, save_debug: bool = False) -> None:
        super().__init__()
        self._executor = executor
        self._save_debug = save_debug
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _enqueue(self, filepath: Path) -> None:
        """Submit the file for processing unless it is already in flight."""
        if not _is_image(filepath):
            return
        name = _book_name(filepath)
        if name is None:
            logger.debug("Ignoring %s – not inside a book subdirectory.", filepath)
            return

        key = str(filepath)
        with self._lock:
            if key in self._in_flight:
                logger.debug("Already queued, skipping duplicate event: %s", filepath.name)
                return
            self._in_flight.add(key)

        logger.info("Queued: %s  (book: '%s')", filepath.name, name)
        self._executor.submit(self._run, filepath, name, key)

    def _run(self, filepath: Path, book_name: str, key: str) -> None:
        """Worker: wait for write completion, then run the OCR pipeline."""
        try:
            if not _wait_for_stable(filepath):
                logger.error("Dropping %s – file did not stabilise.", filepath.name)
                return
            process_book_image(filepath, book_name, save_debug=self._save_debug)
        except Exception:
            # process_book_image has its own handler, but guard against anything
            # unexpected (import errors, OS errors outside the pipeline, etc.)
            logger.error("Unhandled exception processing %s", filepath, exc_info=True)
        finally:
            with self._lock:
                self._in_flight.discard(key)

    # ── Watchdog callbacks ───────────────────────────────────────────────────

    def on_created(self, event) -> None:
        """Fired when a new file is created inside the watch tree."""
        if not event.is_directory:
            self._enqueue(Path(event.src_path))

    def on_moved(self, event) -> None:
        """
        Fired when a file is renamed or moved into the watch tree.

        Many scanner applications write to a temp path and then do an
        atomic rename once the write is complete.  In that case on_moved
        is more reliable than on_created because the file is already fully
        written by the time the event fires – the stability wait will return
        immediately after a single poll.
        """
        if not event.is_directory:
            self._enqueue(Path(event.dest_path))


# ---------------------------------------------------------------------------
# Start-up backlog sweep
# ---------------------------------------------------------------------------

def process_backlog(handler: OCREventHandler) -> int:
    """
    Enqueue any image files already sitting in PROCESSING_DIR at start-up.

    This handles images that arrived while the watcher was not running,
    e.g. after a reboot, a crash, or a manual batch-copy of scans.

    Returns the number of files enqueued.
    """
    if not PROCESSING_DIR.exists():
        logger.debug("Processing directory does not exist yet; skipping backlog sweep.")
        return 0

    found = [p for p in PROCESSING_DIR.rglob("*") if p.is_file() and _is_image(p)]
    for p in found:
        logger.info("Backlog: %s", p)
        handler._enqueue(p)

    if found:
        logger.info("Backlog sweep: enqueued %d file(s).", len(found))
    else:
        logger.info("Backlog sweep: no pending images found.")
    return len(found)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watchdog monitor for the Finnish OCR book-digitization pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=2, metavar="N",
        help="Number of parallel OCR worker threads.  "
             "OCR is CPU-bound; keep this at or below the number of physical cores.",
    )
    parser.add_argument(
        "--no-backlog", action="store_true",
        help="Skip the start-up sweep for pre-existing images.",
    )
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Save intermediate preprocessing images to ~/ocr_pipeline/debug/ "
             "for each processed scan. Useful for tuning the pipeline.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level log output.",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Watch directory: %s", PROCESSING_DIR)

    executor = ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="ocr-worker",
    )
    handler = OCREventHandler(executor, save_debug=args.save_debug)

    if not args.no_backlog:
        process_backlog(handler)

    observer = Observer()
    observer.schedule(handler, str(PROCESSING_DIR), recursive=True)
    observer.start()
    logger.info(
        "Watching for new images  (workers=%d) – press Ctrl+C to stop.", args.workers
    )

    # ── Graceful shutdown on SIGINT (Ctrl+C) or SIGTERM (systemd stop) ──────
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("Received signal %d – initiating graceful shutdown …", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        logger.info("Stopping file-system observer …")
        observer.stop()
        observer.join()
        logger.info("Waiting for in-progress OCR jobs to finish …")
        executor.shutdown(wait=True)
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
