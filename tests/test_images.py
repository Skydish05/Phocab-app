import io

import pytest
from PIL import Image

from app.services import images
from app.services.images import ImageError, prepare_upload

ORIENTATION = 0x0112
GPS_IFD = 0x8825


def encode(image: Image.Image, fmt: str = "JPEG", **save_args) -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt, **save_args)
    return out.getvalue()


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def test_large_photo_is_shrunk_to_1024_on_the_long_side():
    result = decode(prepare_upload(encode(Image.new("RGB", (3000, 2000)))))

    assert result.format == "JPEG"
    assert result.size == (1024, 683)


def test_small_photo_is_not_upscaled():
    result = decode(prepare_upload(encode(Image.new("RGB", (400, 300)))))

    assert result.size == (400, 300)


def test_exif_rotation_is_applied():
    exif = Image.Exif()
    exif[ORIENTATION] = 6  # "rotate 90° clockwise to display", as phones write it
    landscape_pixels = encode(Image.new("RGB", (200, 100)), exif=exif)

    result = decode(prepare_upload(landscape_pixels))

    assert result.size == (100, 200)


def test_exif_including_gps_location_is_stripped():
    exif = Image.Exif()
    exif[ORIENTATION] = 1
    exif.get_ifd(GPS_IFD)[2] = (37.0, 33.0, 0.0)  # GPSLatitude
    original = encode(Image.new("RGB", (200, 100)), exif=exif)
    assert decode(original).getexif().get_ifd(GPS_IFD)  # the fixture really has GPS

    result = decode(prepare_upload(original))

    assert not result.getexif()
    assert "exif" not in result.info


@pytest.mark.parametrize(
    "image, fmt",
    [(Image.new("RGBA", (50, 50)), "PNG"), (Image.new("RGB", (50, 50)), "WEBP")],
)
def test_png_and_webp_are_converted_to_jpeg(image: Image.Image, fmt: str):
    assert decode(prepare_upload(encode(image, fmt))).format == "JPEG"


def test_gif_is_rejected():
    with pytest.raises(ImageError, match="Unsupported image type"):
        prepare_upload(encode(Image.new("P", (50, 50)), "GIF"))


@pytest.mark.parametrize("data", [b"not an image", b"", encode(Image.new("RGB", (50, 50)))[:40]])
def test_unreadable_file_is_rejected(data: bytes):
    with pytest.raises(ImageError):
        prepare_upload(data)


def test_image_with_too_many_pixels_is_rejected(monkeypatch: pytest.MonkeyPatch):
    # Lowered limit instead of a real 50 MP file, which would slow the suite.
    monkeypatch.setattr(images, "MAX_PIXELS", 100 * 100 - 1)

    with pytest.raises(ImageError, match="too many pixels"):
        prepare_upload(encode(Image.new("RGB", (100, 100))))
