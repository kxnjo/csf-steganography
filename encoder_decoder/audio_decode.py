import numpy as np
from scipy.io import wavfile
import random
import math

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
# ^^ starting position of payload



# uses a key to shuffle sample indices -> for what?
# reads number of lsb from each sample in shuffled order
# stops when number of extracted bits matches the payload size
def decode_audio(audio_data, num_bits, num_lsb, key, offset=0):
    # number of samples in audio data array
    num_samples = len(audio_data)

    # num_bits -> size of payload
    # num_samples -> total number of samples in audio
    if num_bits > (num_samples - offset) * num_lsb:
        raise ValueError("Payload length too large for audio samples and LSBs")

    # creates a list of sample numbers starting from offset -> skip header samples (we only want the payload sample)
    number_list = list(range(offset, num_samples))  # skip header samples
    # when encoding, the random.seed is used to encode to hide data better
    # so when decoding, we have to use the same way to "reverse" the hidden data

    # ensure randomization always happeni n the same way as long as used same key
    random.seed(key)
    # 
    random.shuffle(number_list)

    bits = []
    bit_idx = 0
    # same thing as audio, take the lsb of the result then put into a new array
    # just that this is the hidden msg
    for index in number_list:
        sample = audio_data[index]
        for l in range(num_lsb):
            if bit_idx >= num_bits:
                break
            bits.append((sample >> l) & 1)
            bit_idx += 1
        if bit_idx >= num_bits:
            break

    return bits

# reads the first few samples to extract 32-bit header (defined when encode)
# bits returned -> how many bits of hidden data are embedded
# reads LSB of samples to reconstruct the payload size 
def extract_header(audio_data, num_bits, num_lsb):
    bits = []
    bit_idx = 0

    # loops through the audio samples 
    for samplevalue in audio_data:
        """
            extracts up to num_lsb bits from the sample
            if lsb = 5 -> range(num_lsb) = (0,5)
            ^^^ meaning -> [0, 1, 2, 3, 4]

            at i = 0: extract bit at position 0 (lsb)
            at i = 1: extract bit at position 1 (from the right)
            at i = 2: extract at position 2....
        """

        for i in range(num_lsb):
            # to break the inner loop
            if bit_idx >= num_bits:
                break
            # samplevalue >> i -> shift bits to the right by i place (NEEDS TO SHIFT BECUZ LSB IS BY PER SAMPLE)
            # & 1 -> isolates the last bit (LSB) of the shifted results 
            """

                eg:
                samplevalue of 6 -> 00000110
                lsb = 3

                when i = 0:
                (6 >> 0) & 1 (AND)
                & 1 -> taking the lsb of the result 
                right shift the bits by 0 places -> no change 
                00000110   ← this is 6 in binary
                00000001   ← this is 1 in binary

                00000011
                & 
                00000001
                -----------
                00000001

                compare each bit column by column -> if both bits are 1, result is 1, otherwise 0
            """
            # because sample value = 0, even if shift is still 0
            # bits -> each value inside is the lsb value of the sample value after shifting & extracting lsb
            bits.append((samplevalue >> i) & 1)
            bit_idx += 1
        # loops for max 32 times -> after 32 then break out
        if bit_idx >= num_bits:
            break
    # header bits
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
    # if audio data is more than 1 channel (stereo)
    if len(audio_data.shape) > 1:
        # if is stereo, then convert to mono -> ensure simplicity 
        audio_data = audio_data[:, 0]  # use first channel ( follow encoding side )
    
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16) # to ensure that audio data type is int16
    audio_data = audio_data.copy()

    
    header_bits = extract_header(audio_data, 32, num_lsb)
    # to convert the bits into int (some will get np.int(16))
    header_bits = [int(b) for b in header_bits]
    # header stores SIZE OF PAYLOAD
    print(f"Header bits extracted: {header_bits[:32]}")

    payload_length_bits = 0
    # header_bits will be 32 so loops for 32
    for i in header_bits:
        print(i)
        # shifts value to the left by 1 bit
        # then add i (OR)
        print(f"payload_length_bits: {payload_length_bits}")
        print(f"after shifting: {payload_length_bits << 1}")
        # update payload length bits after -> will *2
        # lets say i is 1, then payload_length_bits will start updating next loop

        # get length of payload in bits (SIZE)
        payload_length_bits = (payload_length_bits << 1) | i

    # * to find out how many samples required to store 32 bits header
    # ? How many samples do I need to store 32 bits, if each sample can hold num_lsb bits?
    # each sample legnth = num_lsb -> we extracted 
    """
        eg: num_lsb = 5
        means each sample currently holds 5 bits 
        1 sample -> 5 bits
        x samples -> 32 bits -> 32/5 = 6.4 (need to round up to fit) = 7
    """
    header_sample_count = math.ceil(32 / num_lsb)

    # offset as header_sample_count to tell decode to ignore the 7 samples 
    extracted_bits = decode_audio(audio_data, payload_length_bits, num_lsb, key, offset=header_sample_count)

    # * convert list of extracted payload into bytes

    # empty byte array to store result
    decoded_bytes = bytearray()

    # range(start, stop, step) -> start at 0, stop at length of payload, increment by 8 each time
    # ! first iteration grabs bits 0–7, the second grabs 8–15, the third grabs 16–23...
    for i in range(0, len(extracted_bits), 8):
        byte_val = 0
        # converting every 8 bits into numbers
        for j in range(8):
            # ensuring that the loop is still within the bounds 
            if i + j < len(extracted_bits):
                byte_val = (byte_val << 1) | extracted_bits[i + j]
            else:
                byte_val = byte_val << 1
        decoded_bytes.append(byte_val)

    return bytes(decoded_bytes)
