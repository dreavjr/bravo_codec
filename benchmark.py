import os

import numpy as np
import PIL.Image

from bravo_codec import bravo_encode, bravo_decode

TEST_CASES = ('247', '252', '256', '260', '264', '269', '273', '277', '281', '285' )

class_size_total = 0
conf_size_total  = 0
encoded_size_total = 0

root = os.path.join(os.path.dirname(__file__), 'benchmark_data')

for tc in TEST_CASES:
    class_path = os.path.join(root, f'test_{tc}_pred.png')
    conf_path  = os.path.join(root, f'test_{tc}_conf.webp')

    class_image = PIL.Image.open(class_path)
    conf_image  = PIL.Image.open(conf_path)

    class_array = np.asarray(class_image)
    conf_array  = np.asarray(conf_image)[:,:,0]

    # print(list(conf_array[0]))
    # print(list(np.diff(conf_array[0].astype(np.int16))))

    # Get original files sizes
    class_size = os.path.getsize(class_path)
    conf_size  = os.path.getsize(conf_path)

    # Apply compression
    encoded_bytes = bravo_encode(class_array, conf_array, quantize_levels=0)

    # Get compressed file size
    encoded_size = len(encoded_bytes)

    # Apply decompression
    decoded_class_array, decoded_conf_array = bravo_decode(encoded_bytes)

    # Checks if the decoded arrays are equal to the original ones
    assert np.array_equal(class_array, decoded_class_array)
    assert np.array_equal(conf_array, decoded_conf_array)

    # Print results
    def file_size_fmt(size, suffix="B"):
        for unit in ("", "Ki", "Mi", "Gi", "Ti"):
            if abs(size) < 1024.0:
                return f"{size:3.1f}{unit}B"
            size /= 1024.0
        return f"{size:.1f}Pi{suffix}"
    original_size = class_size + conf_size
    print('Test case: ', tc)
    print('Original size: ', original_size, file_size_fmt(original_size))
    print('Encoded size: ', encoded_size, file_size_fmt(encoded_size))
    print('Compression ratio: ', original_size/encoded_size)
    print()

    # Update totals
    class_size_total += class_size
    conf_size_total  += conf_size
    encoded_size_total += encoded_size

print('Total original size: ', class_size_total, file_size_fmt(class_size_total))
print('Total encoded size: ', encoded_size_total, file_size_fmt(encoded_size_total))
print('Total compression ratio: ', class_size_total/encoded_size_total)
