import itertools
import functools
import heapq
from typing import Dict, Iterable, Tuple, TypeVar, Sequence, Optional, Any


T = TypeVar('T')

def huffman_compute_code(samples: Iterable[Tuple[T, int]], delimiter: Optional[Any] = None) -> Dict[T, str]:
    """
    Gets the Huffman code for given samples.

    Parameters
    ----------
    samples : Iterable[Tuple[T, int]])
        Iterable of unique samples and their frequencies. Behavior is undefined if there are duplicate samples.
    delimiter : Any, optional
        Delimiter to be used at the ending of the encoded data. If None, no delimiter is used. The delimiter should, if
        possible, have a value greater than any other sample. Its value must be different from any other sample, and
        comparable to them.

    Returns
    -------
    Dict[T, str]
        Dictionary of Huffman code.
    """
    if len(samples) == 1:
        return {samples[0][0]: '0'}

    # Create a priority queue (heap) with the samples
    heap = [[freq, [sym, ""]] for sym, freq in samples]
    if delimiter is not None:
        heap.append([0, [delimiter, '']])
    heapq.heapify(heap)

    # Build Huffman tree
    while len(heap) > 1:
        lowest0 = heapq.heappop(heap) # least frequent node in the heap
        lowest1 = heapq.heappop(heap) # second least frequent node in the heap
        # Those two nodes are the next to be "penalized" with a longer code...
        for code in lowest0[1:]:
            code[1] = '0' + code[1]
        for code in lowest1[1:]:
            code[1] = '1' + code[1]
        # ...to then be merged into a new node with the combined frequencies
        lowest0[0] += lowest1[0]
        lowest0.extend(lowest1[1:])
        heapq.heappush(heap, lowest0)

    # Create Huffman dictionary from heap
    huffman_dict = {sym: code for sym, code in heap[0][1:]}
    return huffman_dict


def huffman_make_canonical(huffman_dict: Optional[Dict[T, str]],
                           huffman_lengths: Optional[Sequence[Tuple[T, int]]] = None) -> Dict[T, str]:
    """
    Gets the canonical Huffman code for a given Huffman code. The canonical Huffman code is a Huffman code that is
    prefix-free and has the minimum expected code length. It may be used to compress data without transmitting the
    codebook.

    Parameters
    ----------
    huffman_dict : Dict[T, str]
        Dictionary of Huffman code. May be None if huffman_lengths is provided.
    huffman_lengths : Sequence[Tuple[T, str]], optional
        Sequence of symbols and their code lengths. If not provided, the code lengths are computed from the Huffman
        code. If provided, huffman_dict is ignored.

    Returns
    -------
    Dict[T, str]
        Dictionary of canonical Huffman code.
    """
    # Sorts symbols by code length and then by symbol value
    if huffman_lengths is None:
        huffman_lengths = [(symbol, len(code)) for symbol, code in huffman_dict.items()]
    def compare_symbol_length(a, b):
        if a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        elif a[0] is None:
            return 0 if b[0] is None else 1
        elif b[0] is None:
            return 0 if a[0] is None else -1
        elif a[0] < b[0]:
            return -1
        elif a[0] > b[0]:
            return 1
        else:
            return 0
    sorted_by_length = sorted(huffman_lengths, key=functools.cmp_to_key(compare_symbol_length))

    # Assigns canonical codes
    canonical_dict = {}
    code = 0
    length_prev = sorted_by_length[0][1]

    for symbol, length_curr in sorted_by_length:
        # If the old code length is greater, shifts up new code to match the new length
        if length_curr > length_prev:
            code <<= (length_curr - length_prev)
            length_prev = length_curr
        # Left-aligns the code to match the code length
        canonical_code = format(code, 'b').zfill(length_curr)
        canonical_dict[symbol] = canonical_code
        # Increments the code for the next iteration
        code += 1

    return canonical_dict


def huffman_encode_canonical(huffman_dict: Dict[T, str], symbols: Iterable[T]) -> Sequence[int]:
    """
    Encodes a canonical Huffman code into a sequence of integers.

    Parameters
    ----------
    huffman_dict : Dict[T, str]
        Dictionary of canonical Huffman code.
    symbols : Iterable[T]
        Iterable of symbols to be encoded.

    Returns
    -------
    Iterable[int]
        Encoded Huffman code, represented as a sequence of code lengths.
    """
    return tuple(len(huffman_dict[symbol]) for symbol in symbols)


def huffman_decode_canonical(huffman_lengths: Iterable[int], symbols: Iterable[T]) -> Dict[T, str]:
    """
    Decodes a canonical Huffman code from a sequence of integers representing code lengths.

    Parameters
    ----------
    huffman_lengths : Iterable[int]
        Encoded Huffman code, represented as a sequence of code lengths.
    symbols : Iterable[T]
        Iterable of symbols to be encoded. The order of the symbols must match the order of the code lengths.

    Returns
    -------
    Dict[T, str]
        Dictionary of canonical Huffman code.
    """
    huffman_length_tuples = list(zip(symbols, huffman_lengths))
    return huffman_make_canonical(None, huffman_length_tuples)


def huffman_encode_data(data: Iterable[T], huffman_dict: Dict[T, str], delimiter: Optional[Any] = None) -> bytes:
    """
    Encodes data using a Huffman code.

    Parameters
    ----------
    data : Iterable[T]
        Iterable of symbols to be encoded.
    huffman_dict : Dict[T, str]
        Dictionary of Huffman code.
    delimiter : Any, optional
        Delimiter symbol to be used at the ending of the encoded data. If None, no delimiter is used. The delimiter
        must be the same symbol informed during the creation of the Huffman code.

    Returns
    -------
    bytes
        Encoded data.
    """
    if delimiter is not None:
        data = itertools.chain(data, (delimiter,))
    bits = ''.join(huffman_dict[d] for d in data)
    bits_len = len(bits)
    bits_rem = bits_len % 8
    buffer = bytearray()
    for i in range(0, bits_len-bits_rem, 8):
        buffer.append(int(bits[i:i+8], 2))
    if bits_rem != 0:
        # Adding '1's tends to avoid matches, but may still cause the padding be decoded to extra symbols if the stream
        # is not delimited
        buffer.append(int(bits[-bits_rem:] + '1'*(8-bits_rem), 2))
    return bytes(buffer)


def huffman_decode_data(encoded: bytes, huffman_dict: Dict[T, str], delimiter: Optional[Any] = None) -> Sequence[T]:
    """
    Decodes data using a Huffman code.

    Parameters
    ----------
    encoded : bytes
        Encoded data.
    huffman_dict : Dict[T, str]
        Dictionary of Huffman code.
    delimiter : Any, optional
        Delimiter symbol to be used at the ending of the encoded data. If None, no delimiter is used. The delimiter
        must be the same symbol informed during the creation of the Huffman code and in the encoding of the data.

    Returns
    -------
    Sequence[T]
        Decoded data.

    Caveats
    -------
        If data is not delimited, decoding may append extra symbols due to the bits used for padding the final byte.
    """
    huffman_inv = {v: k for k, v in huffman_dict.items()}
    bits = ''.join(f'{byte:08b}' for byte in encoded)
    data = []
    buffer = ''
    for bit in bits:
        buffer += bit
        symbol = huffman_inv.get(buffer, None)
        if symbol is not None:
            if symbol ==  delimiter:
                break
            data.append(symbol)
            buffer = ''
    return data


def huffman_optimize(huffman_dict: Optional[Dict[T, str]]) -> Dict[T, Tuple[bytes, int]]:
    """
    Optimizes a Huffman code by converting the binary numbers represented as strings of characters into bytes.

    Parameters
    ----------
    huffman_dict : Dict[T, str]
        Dictionary of Huffman code.

    Returns
    -------
    Dict[T, Tuple[bytes, int]]
        Dictionary of optimized Huffman code, containing the bytes and the number of bits effectively used in the last
        byte.
    """
    # Converts each group of 8 bits into a byte, padding the last byte with '0's
    huffman_opt = {}
    for symbol, code in huffman_dict.items():
        code_len = len(code)
        code_rem = code_len % 8
        code = code + '0'*(8-code_rem) if code_rem != 0 else code
        huffman_opt[symbol] = (bytes(int(code[i:i+8], 2) for i in range(0, code_len, 8)), code_rem)
    return huffman_opt


def huffman_encode_opt(data: Iterable[T], huffman_opt: Dict[T, Tuple[bytes, int]],
                       delimiter: Optional[Any] = None) -> Iterable[int]:
    """
    Encodes data using an optimized Huffman code.

    Parameters
    ----------
    data : Iterable[T]
        Iterable of symbols to be encoded.
    huffman_opt : Dict[T, Tuple[bytes, int]]
        Dictionary of optimized Huffman code.
    delimiter : Any, optional
        Delimiter symbol to be used at the ending of the encoded data. If None, no delimiter is used. The delimiter
        must be the same symbol informed during the creation of the Huffman code.

    Returns
    -------
    Iterable[int]
        Encoded data.
    """
    if delimiter is not None:
        data = itertools.chain(data, (delimiter,))
    shift = 0
    remainder = 0
    final_suspend = False
    for d in data:
        code, final_bits = huffman_opt[d]
        final_suspend = shift + final_bits < 8 # shift is not enough to "clear" final padding
        for c in code[:-1] if final_suspend else code:
            yield remainder | (c >> shift)
            remainder = c << (8 - shift) & 0xFF
        if final_suspend:
            remainder |= code[-1] >> shift
            shift = shift + final_bits
        else:
            shift = shift + final_bits - 8
    if final_suspend or shift != 0:
        if shift:
            remainder |= (1 << (8-shift)) - 1
        yield remainder


def huffman_decode_opt(data: bytes, huffman_opt: Dict[T, Tuple[bytes, int]],
                       delimiter: Optional[Any] = None) -> Iterable[T]:
    """
    Decodes data using an optimized Huffman code.

    Parameters
    ----------
    data : bytes
        Encoded data.
    huffman_opt : Dict[T, Tuple[bytes, int]]
        Dictionary of optimized Huffman code.
    delimiter : Any, optional
        Delimiter symbol to be used at the ending of the encoded data. If None, no delimiter is used. The delimiter
        must be the same symbol informed during the creation of the Huffman code and in the encoding of the data.

    Returns
    -------
    Iterable[T]
        Decoded data.

    Caveats
    -------
        If data is not delimited, decoding may append extra symbols due to the bits used for padding the final byte.
    """
    # TODO: this is broken
    huffman_inv = {v[0]: k for k, v in huffman_opt.items()}
    shift = 0
    buffer = b''
    for d in data:
        d_shifted = (d << shift) & 0xFF
        bits_total = 8 - shift
        for bits_n in range(bits_total):
            bits_s = 8 - bits_n - 1
            d_masked = (d_shifted >> bits_s) << bits_s
            buffer_candidate = buffer + int(d_masked).to_bytes(1, 'big')
            symbol = huffman_inv.get(buffer_candidate, None)
            if symbol is not None:
                if symbol ==  delimiter:
                    return
                yield symbol
                shift = (shift + bits_n + 1) % 8
                buffer = b''
                break
        buffer = buffer_candidate

# Test the functions
def test_huffman(num_tests=10, delimiter='z', out=None, symbols_n=6, symbols_min=5, symbols_max=50):
    import random

    for num_tests in range(num_tests):
        samples = [(chr(ord('A') + i), random.randint(symbols_min, symbols_max)) for i in range(symbols_n)]

        huffman_code = huffman_compute_code(samples, delimiter=delimiter)
        symbols = sorted(huffman_code.keys())
        encoded = huffman_encode_canonical(huffman_code, symbols)
        decoded = huffman_decode_canonical(encoded, symbols)
        huffman_code = decoded

        if out is not None:
            print('Samples: ', huffman_code, file=out)
            print('Huffman Code: ', huffman_code, file=out)
            print('Canonical Code: ', huffman_make_canonical(huffman_code), file=out)
            print('Encoded Code: ', encoded, file=out)
            print('Decoded Code: ', decoded, file=out)

        data = [s for s,l in samples for i in range(l)]
        random.shuffle(data)
        data = ''.join(data)

        encoded = huffman_encode_data(data, huffman_code, delimiter=delimiter)
        decoded = huffman_decode_data(encoded, huffman_code, delimiter=delimiter)
        decoded = ''.join(decoded)

        if out is not None:
            print('Data: ', data, file=out)
            print('Data/Len: ', len(data), file=out)
            print('Encoded: ', encoded, file=out)
            print('Encoded/Len: ', len(encoded), file=out)
            print('Decoded: ', decoded, file=out)
            print('Decoded/Len: ', len(decoded), file=out)
            print('Compression ratio: ', len(data)/len(encoded), file=out)
            print('Decoded == Data: ', decoded == data, file=out)
            print('Decoded.startswith(Data): ', decoded.startswith(data), file=out)

        if delimiter:
            assert decoded == data, "Decoded data does not match original data"
        else:
            assert decoded.startswith(data), "Decoded data does not match original data (undelimited)"

        huffman_opt = huffman_optimize(huffman_code)
        encoded_opt = bytes(huffman_encode_opt(data, huffman_opt, delimiter=delimiter))
        decoded_opt = huffman_decode_opt(encoded_opt, huffman_opt, delimiter=delimiter)
        decoded_opt = ''.join(decoded_opt)

        if out is not None:
            print('Huffman Opt: ', huffman_opt, file=out)
            print('Encoded: ', encoded_opt, file=out)
            print('Encoded/Len: ', len(encoded_opt), file=out)
            print('Encoded (opt) == Encoded: ', encoded_opt == encoded, file=out)
            print('Decoded: ', decoded_opt, file=out)
            print('Decoded/Len: ', len(decoded_opt), file=out)
            print('Compression ratio: ', len(data)/len(encoded), file=out)
            print('Decoded == Data: ', decoded_opt == data, file=out)
            print('Decoded.startswith(Data): ', decoded.startswith(data), file=out)

        if delimiter:
            assert decoded == data, "Decoded data does not match original data"
        else:
            assert decoded.startswith(data), "Decoded data does not match original data (undelimited)"

    if out is not None:
        print("All tests passed!", file=out)


if __name__ == "__main__":
    import sys
    test_huffman(out=sys.stdout)
