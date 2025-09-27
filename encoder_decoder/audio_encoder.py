import numpy as np
from scipy.io import wavfile
import random
import sys
import os

# --- Functions ---
def file_to_bits(filename):
    """Read payload file and return a list of bits"""
    with open(filename, "rb") as f:
        byte_array = f.read()
    bits = []
    for byte in byte_array:
        for i in range(8):
            bits.append((byte >> (7-i)) & 1)  # MSB first
    return bits

def bits_to_file(bits, filename):
    """Save bits back to a binary file (for testing)"""
    bytes_out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i+j < len(bits):
                byte = (byte << 1) | bits[i+j]
            else:
                byte = byte << 1
        bytes_out.append(byte)
    with open(filename, "wb") as f:
        f.write(bytes_out)

# --- Create 48-bit header: payload_size (32 bits), start_offset (16 bits) ---
def create_audio_header(payload_bits_len, start_offset):
    header_bits = []
    # 32-bit payload size
    for i in range(32):
        header_bits.append((payload_bits_len >> (31 - i)) & 1)
    # 16-bit start position
    for i in range(16):
        header_bits.append((start_offset >> (15 - i)) & 1)
    return header_bits


def embed_header(audio_data, header_bits, num_lsb):
    """Embed header bits sequentially into the first samples"""
    bit_idx = 0
    for i in range(len(audio_data)):
        sample = audio_data[i]
        for l in range(num_lsb):
            if bit_idx >= len(header_bits):
                break
            sample = sample & ~(1 << l)
            sample = sample | (header_bits[bit_idx] << l)
            bit_idx += 1
        audio_data[i] = sample
        if bit_idx >= len(header_bits):
            break
    return audio_data

def embed_payload(audio_data, payload_bits, num_lsb, key, offset=0):
    """Embed payload bits into audio samples using shuffled indices and key-based bit positions"""
    print("encoding.....")
    audio_data = audio_data.copy()
    num_samples = len(audio_data)

    if len(payload_bits) > (num_samples - offset) * num_lsb:
        raise ValueError("Payload too large for the chosen LSBs")

    # shuffle sample order based on key
    indices = list(range(offset, num_samples))
    random.seed(key)
    random.shuffle(indices)

    # choose which LSB positions to use inside each sample (based on key)
    bit_positions = list(range(8))
    random.shuffle(bit_positions)
    bit_positions = bit_positions[:num_lsb]

    bit_idx = 0
    for sample_idx in indices:
        sample = audio_data[sample_idx]
        for l in bit_positions:
            if bit_idx >= len(payload_bits):
                break
            sample = sample & ~(1 << l)                  # clear bit
            sample = sample | (payload_bits[bit_idx] << l) # set bit
            bit_idx += 1
        audio_data[sample_idx] = sample
        if bit_idx >= len(payload_bits):
            break

    return audio_data

# --- Main CLI ---
if __name__ == "__main__":
    cover_file = input("Enter cover audio filename (.wav): ")
    payload_file = input("Enter payload filename: ")
    num_lsb = int(input("Enter number of LSBs to use (1-8): "))
    key = int(input("Enter key (integer): "))

    if not os.path.isfile(cover_file) or not os.path.isfile(payload_file):
        print("Cover or payload file does not exist.")
        sys.exit(1)

    # Read cover audio
    samplerate, audio_data = wavfile.read(cover_file)
    # Convert to mutable integer array if needed
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    audio_data = audio_data.copy()

    # Read payload
    payload_bits = file_to_bits(payload_file)

    # Embed
    stego_data = embed_payload(audio_data, payload_bits, num_lsb, key)

    # Save stego audio
    wavfile.write("stego_audio.wav", samplerate, stego_data)
    print("Stego audio saved as 'stego_audio.wav'.")
