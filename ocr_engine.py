#!/usr/bin/env python3
"""
ocr_engine.py – Finnish book digitization OCR pipeline module.

Intended to be called by a Watchdog file-system monitor whenever a new
raw scan lands in ~/ocr_pipeline/processing/{book_name}/.

Typical usage:
    from ocr_engine import process_book_image
    success = process_book_image("/abs/path/to/scan.tif", "kalevala")

Expected directory layout under ~/ocr_pipeline/:
    processing/{book_name}/       ← Watchdog drops incoming scans here
    completed/{book_name}.txt     ← Master text file (pages appended in order)
    completed/{book_name}/images/ ← Successfully processed scans archived here
    errors/                       ← Quarantined scans that failed processing
"""

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pytesseract


# ---------------------------------------------------------------------------
# Logging
# All output goes to stderr so it never pollutes stdout-based IPC with Watchdog.
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
OCR_PIPELINE_ROOT = Path.home() / "ocr_pipeline"
COMPLETED_DIR     = OCR_PIPELINE_ROOT / "completed"
ERRORS_DIR        = OCR_PIPELINE_ROOT / "errors"

# OEM 1  = LSTM neural-net engine (most accurate with modern Tesseract ≥ 4).
# PSM 6  = Assume a single uniform block of text – best for full book pages.
# PSM 3 (fully automatic) is used as a fallback for pages with mixed layouts.
TESS_LANG    = "fin"
TESS_CONFIG  = "--oem 1 --psm 6"

# Skew correction is capped at this value to avoid accidentally rotating
# a portrait page into landscape orientation.
MAX_SKEW_DEGREES = 15.0


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert any OpenCV image to a single-channel grayscale array.
    Handles BGR, BGRA (PNG with alpha), and images already in grayscale.
    """
    if img.ndim == 2:
        return img  # Already single-channel
    if img.shape[2] == 4:
        # Drop the alpha channel before converting (avoids a black-background artefact)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Apply a mild Gaussian blur to suppress scanner sensor noise.

    A 3×3 kernel is used deliberately: enough to smooth single-pixel noise
    without blurring the fine strokes of small-point Finnish book typography.
    """
    return cv2.GaussianBlur(gray, (3, 3), sigmaX=0)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct document skew using the minimum-area bounding
    rectangle of all detected text pixels.

    Algorithm:
      1. Produce a temporary binary image (THRESH_BINARY_INV) so text pixels
         are white and easy to collect as point coordinates.
      2. Feed those coordinates to cv2.minAreaRect, which returns the angle of
         the smallest enclosing rectangle – a reliable proxy for page tilt.
      3. Normalise the angle from OpenCV's (-90, 0] range into a signed
         correction value, then rotate the *grayscale* (not the binary) image
         so that the following binarization step still has full tonal range.

    The rotation is only applied if the detected skew is within ±MAX_SKEW_DEGREES
    to guard against mis-classifying a correctly-oriented portrait page.
    """
    # Temporary threshold to locate text pixels (result is never written to disk)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Collect (row, col) coordinates of all text (white) pixels
    yx = np.column_stack(np.where(binary > 0))
    if len(yx) < 100:
        # Too few pixels to estimate a reliable angle – skip deskew
        logger.debug("Skipping deskew: only %d text pixels detected.", len(yx))
        return gray

    # cv2.minAreaRect expects points as (x, y) = (col, row)
    xy = yx[:, ::-1].astype(np.float32)
    raw_angle = cv2.minAreaRect(xy)[-1]  # Returned in range (-90, 0]

    # Convert OpenCV's ambiguous angle to a signed correction angle in (-45, 45]:
    #   raw_angle close to  0  → text tilted slightly clockwise   → correct CCW
    #   raw_angle close to -90 → rectangle is reported via its short axis
    #                            → add 90 to recover the true small skew
    if raw_angle < -45.0:
        correction = -(90.0 + raw_angle)
    else:
        correction = -raw_angle

    if abs(correction) > MAX_SKEW_DEGREES:
        logger.debug(
            "Deskew angle %.2f° exceeds safety limit (%.1f°); skipping.",
            correction, MAX_SKEW_DEGREES,
        )
        return gray

    logger.debug("Deskewing by %.2f°.", correction)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), correction, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,  # Fill rotation border with edge pixels
    )


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Convert the grayscale page to a crisp black-text-on-white-background binary
    image using Otsu's global thresholding.

    Otsu works excellently for evenly-lit scans because it automatically finds
    the optimal threshold between the ink and paper intensity distributions.
    For pages with uneven lighting (curved bindings, yellowed paper, shadows),
    Adaptive (Gaussian-weighted local) thresholding is used as a fallback.

    The fallback is triggered when Otsu yields fewer than 2% or more than 60%
    black pixels, which indicates the global threshold landed on the wrong mode.
    """
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    black_ratio = float(np.count_nonzero(otsu == 0)) / otsu.size

    if 0.02 <= black_ratio <= 0.60:
        return otsu

    logger.debug(
        "Otsu black-pixel ratio %.3f is out of the acceptable range [0.02, 0.60]; "
        "switching to adaptive threshold.", black_ratio
    )
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # Neighbourhood size; ~1% of a 300 DPI A4 page width
        C=15,           # Constant subtracted from local mean – tuned for aged paper
    )


def _preprocess(img: np.ndarray) -> np.ndarray:
    """
    Full pre-processing pipeline delivered as a single call.
    Returns a binary (uint8, values 0 or 255) image ready for Tesseract.
    """
    gray     = _to_grayscale(img)
    denoised = _denoise(gray)
    deskewed = _deskew(denoised)
    binary   = _binarize(deskewed)
    return binary


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _run_ocr(image: np.ndarray) -> str:
    """
    Pass the pre-processed binary image directly to Tesseract in memory.
    No intermediate file is written to disk.
    """
    return pytesseract.image_to_string(
        image,
        lang=TESS_LANG,
        config=TESS_CONFIG,
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Finnish end-of-line hyphen (tavuviiva): "sana-\n  osa" → "sanaosa".
# The \s* around the newline tolerates trailing spaces that Tesseract sometimes adds.
_EOL_HYPHEN_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")

# Isolated single-character tokens that are almost certainly ink speckles or
# ringing artefacts.  We keep: word characters (\w covers letters & digits),
# common Finnish punctuation, brackets, quotation marks, and dash variants.
_JUNK_SINGLE_RE = re.compile(
    r"^[^\w.,;:!?()\[\]{}'\"»«\-–—/\\@#%&*+=<>]$",
    re.UNICODE,
)

# Two or more horizontal whitespace characters on the same line
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Tesseract sometimes inserts a space before closing punctuation: "asia ." → "asia."
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.,;:!?»\)\]])")

# Tesseract sometimes inserts a space after an opening bracket: "( teksti" → "(teksti"
_SPACE_AFTER_OPEN_RE = re.compile(r"([\(«\[]) +")

# Three or more consecutive blank lines → two (preserves paragraph structure)
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")


def _clean_text(raw: str) -> str:
    """
    Repair the most common Finnish OCR artefacts without a dictionary lookup.

    Steps (order is significant):
      1. Merge end-of-line hyphens *before* any line-level work so the joined
         word is presented as a whole token in subsequent passes.
      2. Strip isolated junk tokens (speckle artefacts rendered as stray chars).
      3. Normalise intra-line whitespace.
      4. Fix punctuation spacing errors introduced by Tesseract.
      5. Collapse excessive blank lines while preserving paragraph breaks.
    """
    # 1. Tavuviiva (soft hyphen) merging
    text = _EOL_HYPHEN_RE.sub(r"\1\2", raw)

    # 2. Per-line junk-token removal
    cleaned_lines = []
    for line in text.splitlines():
        tokens = line.split()
        kept = [
            tok for tok in tokens
            if not (len(tok) == 1 and _JUNK_SINGLE_RE.match(tok))
        ]
        cleaned_lines.append(" ".join(kept))
    text = "\n".join(cleaned_lines)

    # 3. Collapse multiple spaces / tabs within a line
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 4. Punctuation spacing
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)

    # 5. Blank-line normalisation
    text = _EXCESS_BLANK_RE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# File-management helpers
# ---------------------------------------------------------------------------

def _append_to_master(book_name: str, page_text: str, source_name: str) -> None:
    """
    Append one page's cleaned text to the book's master .txt file.
    Each page is preceded by a clearly visible separator that names the source
    scan file, making it easy to trace any OCR problem back to its origin image.
    """
    master_path = COMPLETED_DIR / f"{book_name}.txt"
    separator = (
        f"\n\n{'─' * 72}\n"
        f"[PAGE: {source_name}]\n"
        f"{'─' * 72}\n\n"
    )
    with master_path.open("a", encoding="utf-8") as fh:
        fh.write(separator)
        fh.write(page_text)
        fh.write("\n")

    logger.info("Appended %d chars to %s.", len(page_text), master_path.name)


def _archive_image(filepath: Path, book_name: str) -> None:
    """
    Move a successfully processed scan to completed/{book_name}/images/.
    This keeps the processing/ folder clean for Watchdog.
    """
    archive_dir = COMPLETED_DIR / book_name / "images"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / filepath.name
    shutil.move(str(filepath), str(dest))
    logger.info("Archived %s → %s", filepath.name, dest)


def _quarantine_image(filepath: Path) -> None:
    """
    Move a failing scan to errors/ for later manual inspection.
    Appends the inode number to the filename on name collisions so no
    pre-existing error file is silently overwritten.
    """
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ERRORS_DIR / filepath.name
    if dest.exists():
        dest = ERRORS_DIR / f"{filepath.stem}_{filepath.stat().st_ino}{filepath.suffix}"
    shutil.move(str(filepath), str(dest))
    logger.error("Quarantined failing image → %s", dest)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_book_image(
    filepath: Union[str, Path],
    book_name: str,
) -> bool:
    """
    Run the complete OCR pipeline for one book-page scan.

    Pipeline stages
    ---------------
    1. Load image from disk via OpenCV.
    2. Pre-process: grayscale → Gaussian denoise → deskew → Otsu binarization.
    3. OCR via PyTesseract (Finnish language pack, PSM 6, LSTM engine).
    4. Post-process: merge hyphens, strip junk tokens, fix punctuation spacing.
    5. Append cleaned text to ~/ocr_pipeline/completed/{book_name}.txt.
    6. Archive the original scan to completed/{book_name}/images/.

    On any failure, the error is logged to stderr and the image is moved to
    ~/ocr_pipeline/errors/ for manual inspection.

    Parameters
    ----------
    filepath  : Absolute path to the raw scan image (any format OpenCV supports:
                TIFF, JPEG, PNG, BMP, etc.).
    book_name : Logical identifier for the book.  Used as the base name of the
                master text file and archive subdirectory.

    Returns
    -------
    bool
        True if all stages completed successfully, False otherwise.
    """
    filepath = Path(filepath).resolve()
    logger.info("─── OCR start: %s  (book: '%s') ───", filepath.name, book_name)

    # Ensure output directories exist before we do any work
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # ── Stage 1: Load ────────────────────────────────────────────────────
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(
                f"cv2.imread returned None – file missing or unsupported format: {filepath}"
            )
        logger.debug("Loaded image: %s  shape=%s  dtype=%s", filepath.name, img.shape, img.dtype)

        # ── Stage 2: Pre-process ─────────────────────────────────────────────
        processed = _preprocess(img)

        # ── Stage 3: OCR ─────────────────────────────────────────────────────
        raw_text = _run_ocr(processed)
        logger.debug("Raw OCR output: %d characters.", len(raw_text))

        # ── Stage 4: Post-process ────────────────────────────────────────────
        clean_text = _clean_text(raw_text)
        logger.debug("Cleaned text:   %d characters.", len(clean_text))

        # ── Stage 5 & 6: Persist and archive ────────────────────────────────
        _append_to_master(book_name, clean_text, filepath.name)
        _archive_image(filepath, book_name)

        logger.info("─── OCR done:  %s ───", filepath.name)
        return True

    except Exception:
        logger.error(
            "Pipeline failed for '%s'.", filepath,
            exc_info=True,    # Logs full traceback to stderr
        )
        if filepath.exists():
            try:
                _quarantine_image(filepath)
            except Exception:
                logger.error(
                    "Could not quarantine '%s' – file left in place.", filepath,
                    exc_info=True,
                )
        return False


# ---------------------------------------------------------------------------
# Quick manual test – run this file directly with an image path as an argument
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manually test the OCR pipeline on a single image."
    )
    parser.add_argument("image_path", help="Absolute path to the scan image.")
    parser.add_argument("book_name", help="Logical book name (e.g. 'kalevala').")
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG-level logging."
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ok = process_book_image(args.image_path, args.book_name)
    sys.exit(0 if ok else 1)
