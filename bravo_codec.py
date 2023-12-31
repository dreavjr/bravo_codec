"""
Bravo CODEC: Compression scheme for segmentation maps with confidence values

This module provides functionalities for compressing and decompressing 2D arrays
representing segmentation maps with confidence values. The input arrays are
assumed to be 2D arrays. The scheme is intended to be used on the BRAVO Challenge.

The main functions in this module are `bravo_encode` and `bravo_decode`.

Functions
---------
- `bravo_encode(class_array, confidence_array, ...)`
    Encode a 2D array of class labels and a 2D array of confidence values into
    a compressed byte-string.

- `bravo_decode(encoded_bytes)`
    Decode a BRAVO compressed byte-string back into a 2D array of class labels
    and a 2D array of confidence values.

Usage
-----
    from bravo_codec import bravo_encode, bravo_decode

    class_array, confidence_array = your_segmentation_method(input_image)

    # Encoding
    encoded_bytes = bravo_encode(class_array, confidence_array)

    # Decoding
    decoded_class_array, decoded_confidence_array = bravo_decode(encoded_bytes)

Notes
-----
- The class_array compression is lossless.
- The confidence_array compression is lossy, but the loss is controlled by the
  quantization parameters. Use the default values for the BRAVO Challenge.
"""
from collections import Counter
import struct
from typing import Tuple, Union
import zlib

import numpy as np

from rle import run_length_encode2, run_length_decode2
from huffman import huffman_compute_code, huffman_make_canonical, huffman_encode_canonical, huffman_decode_canonical, \
                    huffman_encode_data, huffman_decode_data


HEADER_FORMAT = "<4sIII?III"
HEADER_MAGIC = b"BV23"
COMPRESS_LEVEL = 9 # Compression level for zlib, from 1 to 9, -1 for default


def _compress(data:bytes) -> bytes:
    # Technique 1: zlib compression
    data = zlib.compress(data, level=COMPRESS_LEVEL)
    return data

    # Technique 2: run-length encoding
    # data = run_length_encode2(data)
    # return data

    # Technique 3: run-length encoding + huffman encoding
    # freqs = Counter(data).most_common(256)
    # symbols = np.arange(256, dtype=np.uint8)
    # huffman_table = huffman_compute_code(freqs)
    # huffman_table = huffman_make_canonical(huffman_table)
    # assert len(huffman_table) == 256
    # data = huffman_encode_data(data, huffman_table)
    # huffman_table = huffman_encode_canonical(huffman_table, symbols)
    # huffman_table = bytes(huffman_table)
    # return huffman_table + data

def _decompress(data:bytes) -> bytes:
    # Technique 1: zlib compression
    data = zlib.decompress(data)
    return data

    # Technique 2: run-length encoding
    # data = run_length_decode2(data)
    # return data

    # Technique 3: run-length encoding + huffman encoding
    # huffman_table = data[:256]
    # symbols = np.arange(256, dtype=np.uint8)
    # huffman_table = huffman_decode_canonical(huffman_table, symbols)
    # data = data[256:]
    # data = huffman_decode_data(data, huffman_table)
    # data = bytes(data)
    # data = run_length_decode2(data)
    # return data


def bravo_encode(class_array:np.ndarray[np.uint8],
                  confidence_array:Union[np.ndarray[np.floating],np.ndarray[np.uint8]],
                  quantize_levels:int=100,
                  quantize_linear:bool=True,
                  quantize_n_classes:int=19) -> bytes:
    """
    Encode a class array and confidence array into a BRAVO compressed byte-string.

    Parameters
    ----------
    class_array : np.ndarray[np.uint8]
        Array with class labels. Must be 2D.
    confidence_array : np.ndarray[np.floating] or np.ndarray[np.uint8]
        Array with confidence values for the chosen class. Must be 2D.
    quantize_levels : int, optional
        Number of levels to quantize confidence values to, by default 100. If quantize_levels == 0, confidence_array
        is assumed to be pre-quantized and of type np.uint8
    quantize_linear : bool, optional
        If True, quantize confidence values linearly, otherwise quantize on a logit scale, by default False
    quantize_n_classes : int, optional
        When using the logit scale, the number of classes used to establish the logit = 0 point, by default 19
        Ignored if quantize_linear=True

    Returns
    -------
    bytes
        Compressed byte-string
    """
    # Checks input
    if class_array.ndim != 2:
        raise ValueError("class_array must be 2D")
    if class_array.dtype != np.uint8:
        raise ValueError("class_array must be of dtype np.uint8")
    if confidence_array.ndim != 2:
        raise ValueError("confidence_array must be 2D")
    if class_array.shape != confidence_array.shape:
        raise ValueError("class_array and confidence_array must have the same shape")

    image_shape = class_array.shape

    if confidence_array.dtype == np.uint8:
        if quantize_levels != 0:
            raise ValueError("quantize_levels must be 0 if confidence_array.dtype == np.uint8")
    else:
        if quantize_levels < 2 or quantize_levels > 256:
            raise ValueError("quantize_levels must be between 2 and 256 ")
        if quantize_linear:
            if np.max(confidence_array) > 1. or np.min(confidence_array) < 0.:
                raise ValueError("confidence values must be between 0 and 1 (inclusive)")
        else:
            if np.max(confidence_array) >= 1. or np.min(confidence_array) <= 0.:
                raise ValueError("confidence values must be between 0 and 1 (exclusive)")

    # Linearizes both arrays
    class_array = class_array.ravel()
    confidence_array = confidence_array.ravel()

    if quantize_levels > 0:
        # Linearizes confidence array
        if not quantize_linear:
            # Converts to logit scale, adjusting for number of classes
            # Log( (p/(1-p)) / ((1/n)/(1-1/n)) )
            confidence_array = np.log2(
                    (confidence_array / (1. - confidence_array)) /
                    ((1. / quantize_n_classes) / (1. - 1. / quantize_n_classes))
                )
        # ...quantizes linearly
        confidence_array = (confidence_array * quantize_levels).round()

    # Computes the signed difference between consecutive values
    confidence_array = confidence_array.astype(np.int16)
    confidence_diff = np.diff(confidence_array).astype(np.int8)

    # Compresses both arrays
    class_bytes = class_array.tobytes()
    confidence_bytes = confidence_diff.tobytes()
    data = class_bytes + confidence_bytes
    data = _compress(data)

    # Assembles the header with struct
    header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC,
            image_shape[0],
            image_shape[1],
            quantize_levels,
            quantize_linear,
            confidence_array[0],
            len(class_bytes),
            len(confidence_bytes)
        )

    data = header + data
    crc32 = zlib.crc32(data)

    # Returns the compressed byte-string
    return data + struct.pack("<I", crc32)


def bravo_decode(encoded_bytes: bytes) -> Tuple[np.ndarray[np.uint8], np.ndarray]:
    """
    Decode a BRAVO compressed byte-string into a class array and confidence array.

    Parameters
    ----------
    encoded_bytes : bytes
        The compressed byte-string.

    Returns
    -------
    tuple of np.ndarray[np.uint8] and np.ndarray
        A tuple containing the class array and confidence array. The confidence array is restored to np.float32 if
        the original value of quantize_levels > 0, or kept at np.uint8 otherwise.
    """

    # Parse the header
    header_size = struct.calcsize(HEADER_FORMAT)
    header_bytes = encoded_bytes[:header_size]
    header = struct.unpack(HEADER_FORMAT, header_bytes)
    signature, rows, cols, quantize_levels, quantize_linear, first_confidence, class_len, confidence_len = header

    # Check the signature
    if signature != HEADER_MAGIC:
        raise ValueError("Invalid magic number in header")

    # Check the CRC32
    crc32 = struct.unpack("<I", encoded_bytes[-4:])[0]
    crc32_check = zlib.crc32(encoded_bytes[:-4])
    if crc32 != crc32_check:
        raise ValueError("CRC32 check failed")

    # Decompress the class and confidence arrays
    data = _decompress(encoded_bytes[header_size:-4])
    if len(data) != class_len + confidence_len:
        raise ValueError("Invalid lengths in header")
    class_bytes = data[:class_len]
    confidence_bytes = data[class_len:]

    # Reconstruct the original arrays
    class_array = np.frombuffer(class_bytes, dtype=np.uint8).reshape((rows, cols))
    confidence_diff = np.frombuffer(confidence_bytes, dtype=np.int8)

    # Reconstruct the original confidence array from the differences
    confidence_array = np.zeros(rows*cols, dtype=np.float32)
    confidence_array[0] = first_confidence
    confidence_array[1:] = confidence_diff
    np.cumsum(confidence_array, out=confidence_array)
    confidence_array = confidence_array.reshape((rows, cols))

    # Dequantize the confidence_array
    if not quantize_linear:
        # Convert from logit scale to probability
        raise NotImplementedError("Logit scale not implemented yet")
    else:
        if quantize_levels > 0:
            confidence_array = confidence_array / quantize_levels

    return class_array, confidence_array


def test_bravo_codec(seed=42, array_shape=(1000, 2000), n_classes=19, n_regions=50, out=None):
    np.random.seed(seed)

    # Creates a random but "realistic" class array with a Voronoi tessellation
    n_rows, n_cols = array_shape
    seeds = np.column_stack([
        np.random.randint(0, n_cols, n_regions),
        np.random.randint(0, n_rows, n_regions)
    ])
    classes = np.random.randint(1, n_classes, n_regions)
    rows = np.arange(n_rows)
    cols = np.arange(n_cols)
    coords = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
    # ...computes the distances of coordinates to each seed and finds the closest one
    distances = np.sum((coords[:, None, :] - seeds[None, :, :])**2, axis=2)
    distances_min = np.argmin(distances, axis=1)
    voronoi = distances_min.reshape(n_rows, n_cols)
    # ...assigns the class to each region
    class_array = classes[voronoi].astype(np.uint8)

    # Generate a somewhat "realistic" confidence array: random but smooth
    confidences = np.random.rand(n_regions)
    confidence_array = confidences[voronoi]
    confidence_array += np.random.normal(0, 0.02, size=confidence_array.shape)
    confidence_array = np.clip(confidence_array, 0., 1.)
    confidence_array = confidence_array.astype(np.float32)

    # Encode the arrays
    quantize_levels = 100
    encoded_bytes = bravo_encode(class_array, confidence_array, quantize_levels=quantize_levels)

    # Print the sizes
    def file_size_fmt(size, suffix="B"):
        for unit in ("", "Ki", "Mi", "Gi", "Ti"):
            if abs(size) < 1024.0:
                return f"{size:3.1f}{unit}B"
            size /= 1024.0
        return f"{size:.1f}Pi{suffix}"

    original_size = class_array.nbytes + confidence_array.nbytes
    raw_size = class_array.nbytes + confidence_array.nbytes/4
    encoded_size = len(encoded_bytes)

    if out is not None:
        print("Original size: ", original_size, file_size_fmt(original_size), file=out)
        print("Raw size: ", raw_size, file_size_fmt(raw_size), file=out)
        print("Encoded size:", encoded_size, file_size_fmt(encoded_size), file=out)
        print("Original/encoded ratio:", original_size / encoded_size, file=out)
        print("Raw/encoded ratio:", raw_size / encoded_size, file=out)

    # Decode the arrays
    decoded_class_array, decoded_confidence_array = bravo_decode(encoded_bytes)

    # Verify that the decoded class array matches the original
    assert np.all(decoded_class_array == class_array), "Class arrays do not match"

    # Verify that the decoded confidence array is close to the original within the quantization tolerance
    tolerance = 1 / quantize_levels
    assert np.allclose(decoded_confidence_array, confidence_array, atol=tolerance), \
            f"Confidence arrays do not match within tolerance of {tolerance}"

    if out is not None:
        print("All tests passed!", file=out)


if __name__ == "__main__":
    import sys
    test_bravo_codec(out=sys.stdout)
