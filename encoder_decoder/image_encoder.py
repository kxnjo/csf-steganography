import os
import random
from typing import List

import numpy as np
from PIL import Image


def get_key_permutation(key: int, num_lsb: int) -> List[int]:
    """Generate a permutation of LSB positions based on the key."""
    random.seed(key)
    positions = list(range(num_lsb))
    random.shuffle(positions)
    return positions


def text_to_binary(text: str) -> str:
    """Convert text to binary string."""
    return ''.join(format(ord(char), '08b') for char in text)


def binary_to_bytes(binary_string: str) -> bytes:
    """Convert binary string to bytes."""
    # Pad the binary string to be divisible by 8
    while len(binary_string) % 8 != 0:
        binary_string += '0'

    # Convert to bytes
    byte_array = bytearray()
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        byte_array.append(int(byte, 2))

    return bytes(byte_array)


def embed_data_in_image(image_array: np.ndarray, data_bits: str, num_lsb: int, start_pos: int, key: int) -> np.ndarray:
    """Embed binary data into image using LSB steganography with 56-bit metadata header.
    - start_pos is stored in the header as a pixel index (not channels).
    - Message Length │ LSB Count │ Start Position │
         (32 bits)   │ (8 bits)  │    (16 bits)   │
    """
    height, width, channels = image_array.shape
    total_channels = height * width * channels

    # Capacity check (account for 56 header bits)
    max_bits = (total_channels - 56) * num_lsb
    if len(data_bits) > max_bits:
        raise ValueError(f"Message too long. Maximum {max_bits} bits can be embedded.")

    flat_image = image_array.flatten()

    # --- Metadata header (56 bits) ---
    length_bits = format(len(data_bits), '032b')   # message length in bits
    num_lsb_bits = format(num_lsb, '08b')          # how many LSBs used
    start_pos_bits = format(start_pos, '016b')     # start position in PIXELS
    header_bits = length_bits + num_lsb_bits + start_pos_bits

    # Embed header sequentially in first 56 flat positions, LSB0 only
    for i, bit in enumerate(header_bits):
        pixel_value = int(flat_image[i])
        pixel_value = (pixel_value & ~1) | int(bit)
        flat_image[i] = np.uint8(pixel_value)

    # --- Payload embedding ---
    lsb_positions = get_key_permutation(key, num_lsb)

    # Ensure payload starts after header
    current_pos = max(start_pos * channels, 56)

    bit_index = 0
    for bit in data_bits:
        if current_pos >= len(flat_image):
            break

        pixel_value = int(flat_image[current_pos])
        lsb_index = bit_index % num_lsb
        target_lsb = lsb_positions[lsb_index]

        pixel_value = (pixel_value & ~(1 << target_lsb)) | (int(bit) << target_lsb)
        flat_image[current_pos] = np.uint8(pixel_value)

        bit_index += 1
        if bit_index % num_lsb == 0:
            current_pos += 1

    return flat_image.reshape(height, width, channels)


def encode_message(cover_path: str, message: str, key_text: str, num_lsb: int, start_pos: int) -> str:
    """Pure encoding function.
    - Validates inputs
    - Loads image, embeds message, saves PNG next to source
    - Returns output file path
    Raises ValueError or OSError on failure.
    """
    if not cover_path:
        raise ValueError("Please select a cover image file.")

    if not message:
        raise ValueError("Please enter a message to encode.")

    if not key_text:
        raise ValueError("Please enter an encryption key.")

    if not os.path.exists(cover_path):
        raise FileNotFoundError("Cover image file not found.")

    # Convert key to integer
    try:
        key = int(key_text)
    except ValueError as exc:
        raise ValueError("Encryption key must be a valid integer.") from exc

    # Load and process the image
    image = Image.open(cover_path)

    # Convert to RGB if necessary (handle different formats)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array for manipulation
    image_array = np.array(image)

    # Convert message to binary
    message_binary = text_to_binary(message)

    # Embed the data
    stego_array = embed_data_in_image(image_array, message_binary, num_lsb, start_pos, key)

    # Convert back to PIL Image
    stego_image = Image.fromarray(stego_array.astype(np.uint8))

    # Save the stego image
    base_name = os.path.splitext(cover_path)[0]
    output_path = f"{base_name}_stego.png"
    stego_image.save(output_path, "PNG")

    return output_path


