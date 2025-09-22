import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _key_perm(key: int, num_lsb: int) -> List[int]:
    """Permutation of LSB positions [0..num_lsb-1] based on key."""
    random.seed(key)
    pos = list(range(num_lsb))
    random.shuffle(pos)
    return pos


def _read_header(flat: np.ndarray) -> Tuple[int, int, int]:
    """
    Read 56 bits from LSB0 of the first 56 bytes.
    Layout: [32 bits msg_len] [8 bits num_lsb] [16 bits start_pos_pixels]
    """
    if flat.size < 56:
        raise ValueError("Image too small to contain 56-bit header.")
    bits = ''.join(str(int(flat[i]) & 1) for i in range(56))
    msg_len = int(bits[0:32], 2)
    num_lsb = int(bits[32:40], 2)
    start_pos = int(bits[40:56], 2)
    if not (1 <= num_lsb <= 8):
        raise ValueError(f"Invalid num_lsb in header: {num_lsb}.")
    return msg_len, num_lsb, start_pos


def _extract_bits(arr: np.ndarray, key: int, msg_len_bits: int, num_lsb: int, start_pos_pixels: int) -> str:
    """Walk bytes from max(start_pos*channels, 56), pulling bits in key-permuted LSB order."""
    h, w, c = arr.shape
    flat = arr.flatten()
    idx = max(start_pos_pixels * c, 56)
    perm = _key_perm(key, num_lsb)

    out_bits = []
    total = flat.size
    while len(out_bits) < msg_len_bits:
        if idx >= total:
            raise ValueError("Unexpected end of image while decoding payload.")
        value = int(flat[idx])
        for k in range(num_lsb):
            if len(out_bits) >= msg_len_bits:
                break
            bit = (value >> perm[k]) & 1
            out_bits.append(str(bit))
        idx += 1
    return ''.join(out_bits)


def _bits_to_bytes(bits: str) -> bytes:
    if not bits:
        return b""
    pad = (-len(bits)) % 8
    if pad:
        bits += "0" * pad
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))


def decode_image(stego_path: str, key_text: str) -> Tuple[bytes, Dict[str, int]]:
    """Return (payload_bytes, metadata). Key must be the same integer used at encode time."""
    if not stego_path or not os.path.exists(stego_path):
        raise FileNotFoundError("Stego image not found.")
    if not key_text:
        raise ValueError("Please provide the decoding key.")
    try:
        key = int(key_text)
    except ValueError as e:
        raise ValueError("Key must be an integer.") from e

    img = Image.open(stego_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)

    msg_len_bits, num_lsb, start_pos = _read_header(arr.flatten())
    payload_bits = _extract_bits(arr, key, msg_len_bits, num_lsb, start_pos)
    payload = _bits_to_bytes(payload_bits)

    meta = {
        "message_len_bits": msg_len_bits,
        "num_lsb": num_lsb,
        "start_pos_pixels": start_pos,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "channels": arr.shape[2],
    }
    return payload, meta


def decode_image_to_text(stego_path: str, key_text: str, encoding="utf-8", errors="replace"):
    data, meta = decode_image(stego_path, key_text)
    return data.decode(encoding, errors=errors), meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python decoding.py <stego_image_path> <key>")
        raise SystemExit(1)
    text, meta = decode_image_to_text(sys.argv[1], sys.argv[2])
    print(meta)
    print(text)
