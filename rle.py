"""
This module provides two sets of functions for Run-Length Encoding (RLE) and decoding of byte data.
The first set (`run_length_encode` and `run_length_decode`) implements traditional RLE, while the
second set (`run_length_encode2` and `run_length_decode2`) implements a modified scheme designed to
handle frequent long runs of non-repeating bytes more efficiently.

Both sets of functions operate on byte-like objects and return encoded or decoded data as bytes.
"""
from typing import Union

def run_length_encode(input_buffer: Union[bytes, bytearray]) -> bytes:
    """
    Perform traditional Run-Length Encoding on a byte-like object.

    Parameters
    ----------
    bytes or bytearray
        input_buffer : The byte data to encode.

    Returns
    -------
    bytes
        The run-length encoded data.
    """
    if not input_buffer:
        return b""
    output = bytearray()
    count = 1
    prev_symb = input_buffer[0]
    for symb in input_buffer[1:]:
        if symb == prev_symb and count < 255:
            count += 1
        else:
            output.append(prev_symb)
            output.append(count)
            count = 1
        prev_symb = symb
    output.append(prev_symb)
    output.append(count)
    return bytes(output)

def run_length_decode(encoded_buffer: Union[bytes, bytearray]) -> bytes:
    """
    Decode a traditional Run-Length Encoded byte-like object.

    Parameters
    ----------
    bytes or bytearray
        encoded_buffer : The run-length encoded data.

    Returns
    -------
    bytes
        The decoded data.
    """
    if not encoded_buffer:
        return b""
    output = bytearray()
    for i in range(0, len(encoded_buffer), 2):
        symb = encoded_buffer[i]
        count = int.from_bytes(encoded_buffer[i+1:i+2], byteorder="big")
        output.extend(symb.to_bytes(1, byteorder="big") * count)
    return bytes(output)

def run_length_encode2(input_buffer: Union[bytes, bytearray]) -> bytes:
    """
    Perform a modified Run-Length Encoding on a byte-like object.
    In this modified scheme, a triplet (a, a, n) would mean that the byte 'a' is repeated 'n' times.
    Any other individual bytes imply a run length of 1.

    Parameters
    ----------
    bytes or bytearray
        input_buffer : The byte data to encode.

    Returns
    -------
    bytes
        The run-length encoded data.
    """
    if not input_buffer:
        return b""
    output = bytearray()
    count = 1
    prev_symb = input_buffer[0]
    for symb in input_buffer[1:]:
        if symb == prev_symb and count < 255:
            count += 1
        else:
            output.append(prev_symb)
            if count > 1:
                output.append(prev_symb)
                output.append(count)
            count = 1
        prev_symb = symb
    output.append(prev_symb)
    if count > 1:
        output.append(prev_symb)
        output.append(count)
    return bytes(output)


def run_length_decode2(encoded_buffer: Union[bytes, bytearray]) -> bytes:
    """
    Decodes a modified Run-Length Encoded byte-like object.
    In this modified scheme, a triplet (a, a, n) would mean that the byte 'a' is repeated 'n' times.
    Any other individual bytes imply a run length of 1.

    Parameters
    ----------
    bytes or bytearray
        encoded_buffer : The run-length encoded data.

    Returns
    -------
    bytes
        The decoded data.
    """
    output = bytearray()
    symbol_iter = iter(encoded_buffer)
    try:
        prev_symb = next(symbol_iter, None)
        while prev_symb is not None:
            symb = next(symbol_iter, None)
            if symb == prev_symb:
                count = next(symbol_iter) # raises StopIteration if no count found
                output.extend(prev_symb.to_bytes(1, byteorder="big") * count)
                prev_symb = next(symbol_iter, None)
            else:
                output.append(prev_symb)
                prev_symb = symb
    except StopIteration as e:
        raise ValueError("Invalid buffer: no count found for last run-encoded symbol") from e
    return bytes(output)


def test_rle(num_tests=10, buffer_size=1000, out=None):
    import random

    def generate_random_buffer(size, unique_elements=5, max_run_length=16, singleton_probability=0.5):
        buffer = bytearray()
        buffer_size = 0
        while buffer_size < size:
            if random.random() < singleton_probability:
                run_length = 1
            else:
                run_length = random.randint(1, min(max_run_length, size - buffer_size))
            buffer_size += run_length
            buffer.extend(random.randint(0, unique_elements-1).to_bytes(1, byteorder="big") * run_length)
        return bytes(buffer)

    for _ in range(num_tests):
        original_data = generate_random_buffer(buffer_size)

        encoded_data = run_length_encode(original_data)
        decoded_data = run_length_decode(encoded_data)
        if out is not None:
            print('Encoded/Len: ', len(encoded_data), file=out)
            print('Decoded/Len: ', len(decoded_data), file=out)
            print('Compression ratio: ', len(original_data)/len(encoded_data), file=out)
        assert original_data == decoded_data, "RLE: Decoded data does not match original data"

        encoded_data = run_length_encode2(original_data)
        decoded_data = run_length_decode2(encoded_data)
        if out is not None:
            print('Encoded/Len: ', len(encoded_data), file=out)
            print('Decoded/Len: ', len(decoded_data), file=out)
            print('Compression ratio: ', len(original_data)/len(encoded_data), file=out)
        assert original_data == decoded_data, "RLE2: Decoded data does not match original data"

    if out is not None:
        print("All tests passed!", file=out)


if __name__ == "__main__":
    import sys
    test_rle(out=sys.stdout)
