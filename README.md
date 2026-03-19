# imagecodecs-zarr3

Zarr v3 codec wrappers for [imagecodecs](https://github.com/cgohlke/imagecodecs).

Registers image codecs (JPEG 2000, JPEG, PNG, WebP, AVIF, etc.) as Zarr v3
`ArrayBytesCodec` classes via entry points. Install the package and the codecs
are automatically available to [zarr-python](https://github.com/zarr-developers/zarr-python) —
no manual registration needed.

Pure Python. Delegates to the compiled `imagecodecs` library for encoding/decoding.

## Installation

```bash
pip install imagecodecs-zarr3
```

## Usage

```python
import zarr

# Just open a Zarr array that uses JPEG 2000 chunks — the codec resolves automatically
arr = zarr.open_array(store, mode="r")
data = arr[:]
```

## Registered codecs

| Codec ID | Format |
|----------|--------|
| `imagecodecs_jpeg2k` | JPEG 2000 |
| `imagecodecs_htj2k` | High-Throughput JPEG 2000 |
| `imagecodecs_jpeg8` | JPEG (8-bit) |
| `imagecodecs_jpegls` | JPEG-LS |
| `imagecodecs_jpegxl` | JPEG XL |
| `imagecodecs_jpegxr` | JPEG XR |
| `imagecodecs_ljpeg` | Lossless JPEG |
| `imagecodecs_png` | PNG |
| `imagecodecs_apng` | Animated PNG |
| `imagecodecs_spng` | PNG (libspng) |
| `imagecodecs_webp` | WebP |
| `imagecodecs_avif` | AVIF |
| `imagecodecs_tiff` | TIFF |
| `imagecodecs_gif` | GIF |
| `imagecodecs_bmp` | BMP |
| `imagecodecs_qoi` | QOI |
| `imagecodecs_heif` | HEIF |

## How it works

Each codec is a Zarr v3 `ArrayBytesCodec` that wraps the corresponding
`imagecodecs` decode/encode function. The codec name in `zarr.json` matches
the [Zarr codec registry](https://zarr.dev/codecs-registry/) identifier.

Edge chunks (smaller than the declared chunk shape) are automatically padded
with the array's `fill_value`.

## License

MIT
