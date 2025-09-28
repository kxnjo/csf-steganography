from PIL import Image
from scipy.io import wavfile
import os
import numpy as np
import cv2

def calculate_cover_capacity(cover_path: str, num_lsb: int) -> int:
    """
    Calculate the maximum number of bits that can be stored in the cover object
    based on the selected number of LSBs.

    Args:
        cover_path: Path to the cover file (image or WAV audio)
        num_lsb: Number of least significant bits to use per channel/sample

    Returns:
        max_bits (int): Maximum number of bits that can be stored,
                        or None if unsupported cover type or error.
    """
    if not cover_path or not os.path.isfile(cover_path):
        return None

    ext = os.path.splitext(cover_path)[1].lower()

    try:
        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            img = Image.open(cover_path)
            width, height = img.size
            channels = len(img.getbands())  # e.g., RGB=3, RGBA=4
            max_bits = width * height * channels * num_lsb

        elif ext == ".wav":
            samplerate, audio_data = wavfile.read(cover_path)
            if audio_data.ndim > 1:
                channels = audio_data.shape[1]
                samples = audio_data.shape[0]
            else:
                channels = 1
                samples = audio_data.shape[0]
            max_bits = samples * channels * num_lsb

        elif ext == ".mp4":
            cap = cv2.VideoCapture(cover_path)
            if not cap.isOpened():
                return None
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            channels = 3  # assuming RGB frames
            max_bits = frames * width * height * channels * num_lsb
            cap.release()

        else:
            return None

        return max_bits

    except Exception as e:
        print(f"[cover_capacity] Error calculating capacity for {cover_path}: {e}")
        return None


def calculate_payload_bits(payload: bytes) -> int:
    """
    Return the payload size in bits.

    Args:
        payload (bytes): Payload data

    Returns:
        int: Number of bits in the payload
    """
    if payload is None:
        return 0
    return len(payload) * 8


def check_fit(cover_path: str, num_lsb: int, payload: bytes) -> dict:
    """
    Check whether the payload fits in the cover object with the given number of LSBs.

    Args:
        cover_path (str): Path to the cover file
        num_lsb (int): Number of LSBs to use
        payload (bytes): Payload data

    Returns:
        dict:
            max_bits (int or None): Maximum bits in the cover
            payload_bits (int): Bits required for payload
            fit (bool): True if payload fits, False otherwise
    """
    max_bits = calculate_cover_capacity(cover_path, num_lsb)
    payload_bits = calculate_payload_bits(payload)
    fit = max_bits is not None and payload_bits <= max_bits

    return {"max_bits": max_bits, "payload_bits": payload_bits, "fit": fit}
