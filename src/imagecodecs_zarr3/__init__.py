"""Zarr v3 codec wrappers for imagecodecs.

Registers image codecs (JPEG 2000, JPEG, PNG, WebP, etc.) from the
imagecodecs library as Zarr v3 ArrayBytesCodec classes. Install this
package and the codecs are automatically available via entry points —
no manual registration needed.

Each codec wraps the corresponding imagecodecs encode/decode function.
The compiled C code lives in imagecodecs; this package is pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import imagecodecs
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer


def _make_codec_class(
    codec_name: str,
    decode_func: Any,
    encode_func: Any | None,
    *,
    default_config: dict[str, Any] | None = None,
) -> type:
    """Generate a Zarr v3 ArrayBytesCodec class for an imagecodecs codec.

    Parameters
    ----------
    codec_name : the codec identifier (e.g. "imagecodecs_jpeg2k")
    decode_func : imagecodecs decode function (e.g. imagecodecs.jpeg2k_decode)
    encode_func : imagecodecs encode function, or None for decode-only codecs
    default_config : default configuration dict for the codec
    """
    _default_config = default_config or {}

    @dataclass(frozen=True)
    class _Codec(ArrayBytesCodec):
        is_fixed_size = False
        configuration: dict[str, Any] = field(default_factory=lambda: dict(_default_config))

        @classmethod
        def from_dict(cls, data: dict) -> Self:
            config = data.get("configuration", {})
            return cls(configuration=config)

        def to_dict(self) -> dict:
            d: dict[str, Any] = {"name": codec_name}
            if self.configuration:
                d["configuration"] = self.configuration
            return d

        async def _decode_single(
            self, chunk_bytes: Buffer, chunk_spec: ArraySpec
        ) -> NDBuffer:
            raw = chunk_bytes.to_bytes()
            decoded = decode_func(raw)
            target_dtype = np.dtype(chunk_spec.dtype.to_native_dtype())
            if decoded.dtype != target_dtype:
                decoded = decoded.astype(target_dtype)
            # Edge chunks may be smaller than chunk_spec.shape — pad
            if decoded.shape != chunk_spec.shape:
                padded = np.full(
                    chunk_spec.shape, chunk_spec.fill_value, dtype=target_dtype
                )
                slices = tuple(slice(0, s) for s in decoded.shape)
                padded[slices] = decoded
                decoded = padded
            return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

        async def _encode_single(
            self, chunk_array: NDBuffer, chunk_spec: ArraySpec
        ) -> Buffer | None:
            if encode_func is None:
                raise NotImplementedError(
                    f"{codec_name} does not support encoding"
                )
            arr = chunk_array.as_numpy_array()
            encoded = encode_func(arr, **self.configuration)
            return chunk_spec.prototype.buffer.from_bytes(encoded)

        def compute_encoded_size(
            self, input_byte_length: int, chunk_spec: ArraySpec
        ) -> int:
            return input_byte_length  # variable size

    _Codec.__name__ = codec_name.replace("imagecodecs_", "").title() + "Codec"
    _Codec.__qualname__ = _Codec.__name__
    return _Codec


def _get_func(name: str) -> Any | None:
    """Get an imagecodecs function by name, returning None if unavailable."""
    return getattr(imagecodecs, name, None)


# ---------------------------------------------------------------------------
# Codec class definitions
# ---------------------------------------------------------------------------

Jpeg2kCodec = _make_codec_class(
    "imagecodecs_jpeg2k",
    imagecodecs.jpeg2k_decode,
    _get_func("jpeg2k_encode"),
)

Htj2kCodec = _make_codec_class(
    "imagecodecs_htj2k",
    imagecodecs.jpeg2k_decode,  # HTJ2K uses same decoder
    _get_func("jpeg2k_encode"),
)

Jpeg8Codec = _make_codec_class(
    "imagecodecs_jpeg8",
    imagecodecs.jpeg8_decode if hasattr(imagecodecs, "jpeg8_decode") else imagecodecs.jpeg_decode,
    _get_func("jpeg8_encode") or _get_func("jpeg_encode"),
)

JpeglsCodec = _make_codec_class(
    "imagecodecs_jpegls",
    imagecodecs.jpegls_decode,
    _get_func("jpegls_encode"),
)

JpegxlCodec = _make_codec_class(
    "imagecodecs_jpegxl",
    imagecodecs.jpegxl_decode,
    _get_func("jpegxl_encode"),
)

JpegxrCodec = _make_codec_class(
    "imagecodecs_jpegxr",
    imagecodecs.jpegxr_decode,
    _get_func("jpegxr_encode"),
)

LjpegCodec = _make_codec_class(
    "imagecodecs_ljpeg",
    imagecodecs.ljpeg_decode,
    _get_func("ljpeg_encode"),
)

PngCodec = _make_codec_class(
    "imagecodecs_png",
    imagecodecs.png_decode,
    _get_func("png_encode"),
)

ApngCodec = _make_codec_class(
    "imagecodecs_apng",
    imagecodecs.apng_decode,
    _get_func("apng_encode"),
)

SpngCodec = _make_codec_class(
    "imagecodecs_spng",
    _get_func("spng_decode") or imagecodecs.png_decode,
    _get_func("spng_encode") or _get_func("png_encode"),
)

WebpCodec = _make_codec_class(
    "imagecodecs_webp",
    imagecodecs.webp_decode,
    _get_func("webp_encode"),
)

AvifCodec = _make_codec_class(
    "imagecodecs_avif",
    imagecodecs.avif_decode,
    _get_func("avif_encode"),
)

TiffCodec = _make_codec_class(
    "imagecodecs_tiff",
    imagecodecs.tiff_decode,
    _get_func("tiff_encode"),
)

GifCodec = _make_codec_class(
    "imagecodecs_gif",
    imagecodecs.gif_decode,
    _get_func("gif_encode"),
)

BmpCodec = _make_codec_class(
    "imagecodecs_bmp",
    imagecodecs.bmp_decode,
    _get_func("bmp_encode"),
)

QoiCodec = _make_codec_class(
    "imagecodecs_qoi",
    imagecodecs.qoi_decode,
    _get_func("qoi_encode"),
)

HeifCodec = _make_codec_class(
    "imagecodecs_heif",
    imagecodecs.heif_decode,
    _get_func("heif_encode"),
)
