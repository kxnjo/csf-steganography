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

# offset -> number of samples to skip (to skip from header embedded and start from payload)

# extracts the actual hidden bits from audio samples
# uses a key to shuffle sample indices -> for what?
# reads number of lsb from each sample in shuffled order
# stops when number of extracted bits matches the payload size
def decode_audio(audio_data, num_bits, num_lsb, key, offset=0):
    num_samples = len(audio_data)
    # print(f"Number of audio samples: {num_samples}")
    # print(f"Total bits available for hiding: {(num_samples - offset) * num_lsb}")
    # print(f"Number of bits in the payload: {num_bits}")

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

# reads the first few samples to extract 32-bit header
# bits returned -> how many bits of hidden data are embedded
# reads LSB of samples to reconstruct the payload size 
def extract_header(audio_data, num_bits, num_lsb):
    bits = []
    bit_idx = 0

    # loops through the audio samples 
    for samplevalue in audio_data:
        # extracts up to num_lsb bits from the sample
        for i in range(num_lsb):
            # to break the inner loop
            if bit_idx >= num_bits:
                break
            # samplevalue >> i -> shift bits to the right by i place 
            # & 1 -> isolates the last bit (LSB) of the shifted results 
            """

                eg:
                samplevalue of 6 -> 00000110
                lsb = 3

                when i = 0:
                (6 >> 0) & 1 
                shift the bits by 0 places -> no change 
                00000110   ← this is 6 in binary
                00000001   ← this is 1 in binary

                compare each bit column by column -> if both bits are 1, result is 1, otherwise 0
            """
            bits.append((samplevalue >> i) & 1)
            bit_idx += 1
        # loops for max 32 times -> after 32 then break out
        if bit_idx >= num_bits:
            break
    return bits


# reads audio file input & selects a channel 
# get the 32-bit header from extract_header func
# calculate number of samples used for the header
# use decode_audio to extract payload bits
# convert extracted bits into bytes (1 byte -> 8 bits)
# returns decoded hidden data as bytes 
def decode_wav_file_gui(stego_path, num_lsb, key):
    """
    audio data: [[ 0  0]
                [ 0  0]
                [ 0  0]
                ...
                [-2 -2]
                [ 0  2]
                [ 2 -2]]
    
    each row -> single audio saample index
    each column -> channel (stream of audio data)
    2 types of channels :
    - mono (1 channel) -> all audio come from single source
    - stereo (2 channels) -> left & right , used to create spatial sound (like headphones)
    
    values (samples) representation -> heght of the sound:
    0 -> silence
    2 (positive value) -> wave is above the 0 line
    -2 (negative value) -> wave is below the 0 line

    the whole audio_data array will represent the values & when plotted, will get the waveform of the audio
    """

    # to read the audio file & return as array of audio samples
    # samplerate -> number of samples per second (hz)
    # audio_data NumPy array containing actual audio samples ( like the one above )
    samplerate, audio_data = wavfile.read(stego_path)

    # len(audio_data) -> number of rows in the audio_data array
    # len(audio_data.shape) -> number of channels 
    # audio_data.shape -> (262094, 2) || (no of rows, no of channels)
    print(f"audio data type: {audio_data}")
    # if audio data is more than 1 channel (stereo)
    if len(audio_data.shape) > 1:
        # if is stereo, then convert to mono -> ensure simplicity 
        audio_data = audio_data[:, 0]  # use first channel ( follow encoding side )
    
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16) # to ensure that audio data type is int16
    audio_data = audio_data.copy()

    # STOP HERE
    header_bits = extract_header(audio_data, 32, num_lsb)
    header_bits = [int(b) for b in header_bits]
    print(f"Header bits extracted: {header_bits[:32]}")

    payload_length_bits = 0
    for bit in header_bits:
        payload_length_bits = (payload_length_bits << 1) | bit
    # print(f"Payload length in bits (extracted from header): {payload_length_bits}")

    # Step 2: Calculate header sample count
    # when encoding, my header is 32 bits 
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
