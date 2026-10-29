"""Validate an uploaded photo and turn it into the one JPEG we store and classify."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_SIDE = 1024
# MPO is how Pillow identifies many iPhone and Android JPEGs (a JPEG with extra
# embedded frames); rejecting it would reject ordinary phone photos.
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
# Checked from the header before decoding: a tiny, highly compressed file can
# decode to gigabytes (a "decompression bomb"). 50 MP is above any phone camera's
# default output. Pillow's own limit is roughly 179 MP, too lax for a 4 GB box.
MAX_PIXELS = 50_000_000


class ImageError(Exception):
    """The upload isn't an image we accept. The API maps this to a 400."""


def prepare_upload(data: bytes) -> bytes:
    """Return an upright RGB JPEG with at most MAX_SIDE px on the long side.

    Re-encoding drops all EXIF metadata, including GPS location, so the stored
    photo can't reveal where the user lives. It also caps what we store and send
    to the model at about 1 MP.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            # The format comes from the file's bytes, not the client's Content-Type.
            if image.format not in ALLOWED_FORMATS:
                raise ImageError("Unsupported image type; upload a JPEG, PNG or WebP")
            if image.width * image.height > MAX_PIXELS:
                raise ImageError("Image has too many pixels")
            # Phone cameras record rotation in EXIF instead of rotating the pixels;
            # apply it before EXIF is dropped, or the photo is stored sideways.
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("File is not a readable image") from exc
    upright.thumbnail((MAX_SIDE, MAX_SIDE))  # keeps aspect ratio, never upscales
    out = io.BytesIO()
    upright.save(out, format="JPEG", quality=85)
    return out.getvalue()
