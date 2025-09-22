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

def embed_payload(cover_audio, payload_bits, num_lsb, key):
    """Embed payload bits into audio samples using key"""
    audio_data = cover_audio.copy()
    num_samples = len(audio_data)
    
    if len(payload_bits) > num_samples * num_lsb:
        raise ValueError("Payload too large for the chosen LSBs")

    # Generate shuffled indices based on key
    indices = list(range(num_samples))
    random.seed(key)
    random.shuffle(indices)

    bit_idx = 0
    for sample_idx in indices:
        sample = audio_data[sample_idx]
        for l in range(num_lsb):
            if bit_idx >= len(payload_bits):
                break
            # Clear LSB l and set it to payload bit
            sample = sample & ~(1 << l)
            sample = sample | (payload_bits[bit_idx] << l)
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
