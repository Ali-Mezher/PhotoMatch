"""In-memory selfie decoding and protected preview rendering."""

from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as a supported image."""


def decode_selfie(upload, max_pixels: int = 25_000_000) -> np.ndarray:
    """Decode a Flask upload into a contiguous BGR array without disk writes."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(upload.stream) as opened:
                if opened.width * opened.height > max_pixels:
                    raise InvalidImageError(
                        "That image has too many pixels. Choose a smaller selfie."
                    )
                image = ImageOps.exif_transpose(opened).convert("RGB")
                rgb = np.asarray(image, dtype=np.uint8)
    except InvalidImageError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidImageError("The selected file is not a readable image.") from exc

    if rgb.size == 0:
        raise InvalidImageError("The selected image is empty.")
    return np.ascontiguousarray(rgb[:, :, ::-1])


def render_watermarked_preview(
    photo_path: Path, max_size: tuple[int, int]
) -> BytesIO:
    """Render a resized, watermarked JPEG without changing the source photo."""
    with Image.open(photo_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(14, min(image.size) // 18)
    font = ImageFont.load_default(size=font_size)
    label = "PHOTOMATCH PREVIEW"
    bounds = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = max(12, (image.width - text_width) // 2)
    y = max(12, (image.height - text_height) // 2)
    draw.rounded_rectangle(
        (x - 14, y - 10, x + text_width + 14, y + text_height + 10),
        radius=6,
        fill=(12, 18, 24, 150),
    )
    draw.text(
        (x, y),
        label,
        font=font,
        fill=(255, 255, 255, 225),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 150),
    )
    protected = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    output = BytesIO()
    protected.save(output, format="JPEG", quality=86, optimize=True)
    output.seek(0)
    return output
