import numpy as np
from scipy.io import wavfile
import random

# ==== IDEA ====    
# to read stego audio file with hidden data
# use same key from encode & num of lsb during encoding to decode
# extract hidden bits from audio samples
# put extracted payload in another file


# key -> controls how the sample indices are shuffled || if wrong, will lead to garbage data 
# determines the order of sample position to hide / extract bits -> hide first bit in sample #172, 2nd in $5....
# controls the hiding pattern

# number of LSB -> tells the decoder how many bits per sample were used to hide data 
# if use fewer than encode, will lead to missing bits
# if use more, will lead to noise

# payload size -> to let the decoder know how many bits to extract
# if too few, the file incomplete
# if too much, will get junk

# --------------------------
# Extract payload using shuffled indices
# --------------------------
def decode_audio(audio_data, num_bits, num_lsb, key, offset=0):
    num_samples = len(audio_data)
    print(f"Number of audio samples: {num_samples}")
    print(f"Total bits available for hiding: {(num_samples - offset) * num_lsb}")
    print(f"Number of bits in the payload: {num_bits}")

    if num_bits > (num_samples - offset) * num_lsb:
        raise ValueError("Payload length too large for audio samples and LSBs")

    indices = list(range(offset, num_samples))  # skip header samples
    random.seed(key)
    random.shuffle(indices)

    bits = []
    bit_idx = 0
    for sample_idx in indices:
        sample = audio_data[sample_idx]
        for l in range(num_lsb):
            if bit_idx >= num_bits:
                break
            bits.append((sample >> l) & 1)
            bit_idx += 1
        if bit_idx >= num_bits:
            break

    return bits

# --------------------------
# Extract header sequentially
# --------------------------
def extract_header(audio_data, num_bits, num_lsb):
    bits = []
    bit_idx = 0
    for sample in audio_data:
        for l in range(num_lsb):
            if bit_idx >= num_bits:
                break
            bits.append((sample >> l) & 1)
            bit_idx += 1
        if bit_idx >= num_bits:
            break
    return bits



def decode_wav_file_gui(stego_path, num_lsb, key):
    print(f"stego path: {stego_path}")
    print(f"num_lsb: {num_lsb}")
    print(f"key: {key}")
    """Wrapper to return decoded bytes instead of writing to file."""
    samplerate, audio_data = wavfile.read(stego_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # use first channel
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    audio_data = audio_data.copy()

    # Step 1: Extract header
    print("\nExtracting header...")
    header_bits = extract_header(audio_data, 32, num_lsb)
    print(f"Header bits extracted: {header_bits[:32]}")

    payload_length_bits = 0
    for bit in header_bits:
        payload_length_bits = (payload_length_bits << 1) | bit
    print(f"Payload length in bits (extracted from header): {payload_length_bits}")

    # Step 2: Calculate header sample count
    header_sample_count = (32 + num_lsb - 1) // num_lsb

    # Step 3: Extract payload
    extracted_bits = decode_audio(audio_data, payload_length_bits, num_lsb, key, offset=header_sample_count)

    # Step 4: Convert bits to bytes
    decoded_bytes = bytearray()
    for i in range(0, len(extracted_bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(extracted_bits):
                byte_val = (byte_val << 1) | extracted_bits[i + j]
            else:
                byte_val = byte_val << 1
        decoded_bytes.append(byte_val)

    return bytes(decoded_bytes)
