# BRAVO CODEC

Reference implementation for the BRAVO Challenge encoder/decoder for the class label + confidence level segmentation masks.

## Overview

The BRAVO CODEC is a specialized compression scheme designed for 2D segmentation maps accompanied by confidence values. It offers an efficient way to store and transmit these kinds of specialized data. The compression is particularly useful for the BRAVO Challenge but can be applied to any project that requires storing segmentation maps with confidence values.

## Features

- Lossless compression for class labels.
- Lossy compression for confidence values, with controllable quantization levels.
- Fast compression and decompression using Python's built-in `zlib` library.
- Portable, with no dependencies other than `numpy`.

## Installation

You can clone this repository to your local machine using:

```bash
git clone https://github.com/yourusername/bravo-codec.git
```

Then, navigate to the folder and you're ready to go.

## Dependencies

- Python 3.x
- NumPy

## Usage

First, import the required functions:

```python
from bravo_codec import bravo_encode, bravo_decode
```

Then, use your segmentation method to get the class array and confidence array.

```python
class_array, confidence_array = your_segmentation_method(input_image)
```

### Encoding

```python
encoded_bytes = bravo_encode(class_array, confidence_array)
```

### Decoding

```python
decoded_class_array, decoded_confidence_array = bravo_decode(encoded_bytes)
```

## Benchmarking

You may run a benchmark of the codec, contrasting its performance on real data with a combo of PNG+WebP:

```bash
python benchmark.py
```

The benchmark has Pillow as an additional dependency.
