import subprocess

from scipy.io import wavfile
import numpy as np

import os

from encoder_decoder.audio_encoder import file_to_bits, embed_payload, embed_header, create_audio_header

def extract_audio(mp4_path, wav_out="cover_audio.wav"):
    subprocess.run([
        "ffmpeg.exe", "-y", "-i", mp4_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", wav_out
    ])

    return wav_out 

def encode_payload_in_audio(wav_path, payload_path, stego_wav_path, num_lsb, key, start_pos):
    samplerate, audio_data = wavfile.read(wav_path)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # mono for simplicity
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)

    payload_bits = file_to_bits(payload_path)
    payload_size = len(payload_bits)

    header_bits = create_audio_header(payload_size, start_pos)
    audio_data = embed_header(audio_data, header_bits, num_lsb)

    # Calculate how many samples were used for header
    header_sample_count = (48 + num_lsb - 1) // num_lsb
    payload_offset = header_sample_count + start_pos
    stego_data = embed_payload(audio_data, payload_bits, num_lsb, key, offset=payload_offset)
    wavfile.write(stego_wav_path, samplerate, stego_data.astype(np.int16))

def combine_audio_video(original_video, stego_audio, output_video="stego_video.mp4"):
    subprocess.run([
        "ffmpeg.exe","-y", "-i", original_video, "-i", stego_audio,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_video
    ])

    return output_video  
