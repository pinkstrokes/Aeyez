from __future__ import annotations

import base64

import pytest

from src.seeingeye.runtime.media import (
    _validate_frame_interval,
    _validate_frame_selection,
    encode_image,
)


def test_encode_image_returns_plain_base64(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image-bytes")

    assert encode_image(image) == base64.b64encode(b"image-bytes").decode("ascii")


@pytest.mark.parametrize("value", [0.1, 0.5, 1.0])
def test_validate_frame_interval_accepts_requested_range(value) -> None:
    assert _validate_frame_interval(value) == value


@pytest.mark.parametrize("value", [0.09, 1.01])
def test_validate_frame_interval_rejects_outside_requested_range(value) -> None:
    with pytest.raises(ValueError, match="between 0.1 and 1.0"):
        _validate_frame_interval(value)


@pytest.mark.parametrize("value", ["uniform", "change", " CHANGE "])
def test_validate_frame_selection_accepts_known_modes(value) -> None:
    assert _validate_frame_selection(value) == value.strip().lower()


def test_validate_frame_selection_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="uniform.*change"):
        _validate_frame_selection("random")
