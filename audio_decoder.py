# audio_decode.py
import numpy as np
from scipy.io import wavfile
import random

def extract_payload(stego_audio, num_lsb, key):
    num_samples = len(stego_audio)

    indices = list(range(num_samples))
    random.seed(key)
    random.shuffle(indices)

    # Read first 32 bits (payload length)
    bits = []
    bit_idx = 0
    for sample_idx in indices:
        sample = stego_audio[sample_idx]
        for l in range(num_lsb):
            bits.append((sample >> l) & 1)
            bit_idx += 1
            if bit_idx >= 32:
                break
        if bit_idx >= 32:
            break

    payload_len = 0
    for b in bits:
        payload_len = (payload_len << 1) | b

    # Extract payload bits
    bits = []
    bit_idx = 0
    read_bits = 0
    for sample_idx in indices:
        sample = stego_audio[sample_idx]
        for l in range(num_lsb):
            if read_bits < 32:
                read_bits += 1
                continue  # skip header bits
            if bit_idx >= payload_len:
                break
            bits.append((sample >> l) & 1)
            bit_idx += 1
            read_bits += 1
        if bit_idx >= payload_len:
            break

    return bits

def bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
            else:
                byte <<= 1
        out.append(byte)
    return bytes(out)

def decode_wav_file(stego_file, num_lsb, key):
    samplerate, stego_data = wavfile.read(stego_file)

    # Convert stereo to mono if needed
    if len(stego_data.shape) > 1:
        stego_data = stego_data[:, 0]

    if stego_data.dtype != np.int16:
        stego_data = stego_data.astype(np.int16)

    bits = extract_payload(stego_data, num_lsb, key)
    data = bits_to_bytes(bits)
    return data

