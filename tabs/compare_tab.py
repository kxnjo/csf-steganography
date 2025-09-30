import customtkinter as ctk
from assets.dragdrop import DragDropLabel
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile  # Fixed import
import wave
import io
import os

from assets.ScrollableFrame import ScrollableFrame  # custom scrollable frame

# for image analysis
from scipy.stats import chisquare
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_wavelet
import math

# for audio analysis
import numpy.fft as fft  # phase coding detection
from scipy.signal import correlate  # echo detection

# import similarity comparison (for final suspicion)
from skimage.metrics import structural_similarity as ssim

# detect if it is an image (or exceptable file forms)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")


# = = = IMAGE COMPARISON = = =
# --- histogram ---
def show_histogram(original, suspect, result_label, bins=256):
    """Works for both image (2D) and audio (1D) arrays"""
    # flatten in case it's 2D (image)
    orig_flat = original.ravel()
    sus_flat = suspect.ravel()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=100)

    ax[0].hist(orig_flat, bins=bins, color="blue", alpha=0.7)
    ax[0].set_title("Original Histogram")

    ax[1].hist(sus_flat, bins=bins, color="red", alpha=0.7)
    ax[1].set_title("Suspect Histogram")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(500, 250))
    result_label.configure(image=img_ctk, text="")
    result_label.image = img_ctk


# --- show lsb ---
def show_lsb_map(original, suspect, result_label):
    # extract least significant bit (LSB) plane
    orig_lsb = original & 1
    suspect_lsb = suspect & 1

    # XOR to highlight differences
    diff_map = (orig_lsb ^ suspect_lsb) * 255

    diff_img = Image.fromarray(diff_map.astype(np.uint8))
    img_ctk = ctk.CTkImage(light_image=diff_img, dark_image=diff_img, size=(400, 300))

    result_label.configure(image=img_ctk, text="")
    result_label.image = img_ctk


# --- to view stats ---
# entropy: show how random it the pixels are laid.
# Cover image (original): has a natural entropy depending on its content.
# Stego image (suspect): hiding data usually introduces extra randomness (changes in least significant bits).
# If the entropy increases noticeably → it suggests that “hidden information” was embedded.
# If they’re nearly identical → little or no evidence of embedding.


def create_stat_card(parent, title, value, subtitle="", color="white"):
    card = ctk.CTkFrame(parent, corner_radius=10, fg_color="gray25")
    card.pack(side="left", expand=True, fill="both", padx=5, pady=5)

    # Title
    title_label = ctk.CTkLabel(card, text=title, font=("Arial", 14, "bold"))
    title_label.pack(anchor="w", padx=10, pady=(10, 0))

    # Value
    value_label = ctk.CTkLabel(
        card, text=value, font=("Arial", 24, "bold"), text_color=color
    )
    value_label.pack(anchor="w", padx=10, pady=(5, 0))

    # Subtitle
    if subtitle:
        subtitle_label = ctk.CTkLabel(
            card, text=subtitle, font=("Arial", 12), text_color="lightgray"
        )
        subtitle_label.pack(anchor="w", padx=10, pady=(0, 10))

    return card


def show_stats(original, suspect, container, mode="image"):
    # Clear old cards row
    for w in container.winfo_children():
        w.destroy()
    row = ctk.CTkFrame(container, fg_color="transparent")
    row.pack(fill="x", pady=5)

    if mode == "image":
        # Display the percentage of pixel difference
        diff = np.abs(suspect - original)
        changed_pixels = int(np.count_nonzero(diff))
        total_pixels = int(diff.size)
        diff_percent = (changed_pixels / max(1,total_pixels)) * 100

        # shannon entropy: uncertainty or randomness in information
        # the higher the value = more uncertainty = harder to predict = more likely to have changed
        # the lower the value = more predictable = lesser changes
        orig_entropy = shannon_entropy(original)
        suspect_entropy = shannon_entropy(suspect)

        # Chi-Square Test χ² (image bins = image categories (1 or 0) values )
        # checks whether the observed frequencies in the bins deviate significantly from what’s expected.
        # detects "uniform"/"randomness" in each pixel
        image_bins = 64 # for unnatural pixel distribution
        ho, _ = np.histogram(original, bins=image_bins, range=(0,255))
        hs, _ = np.histogram(suspect,  bins=image_bins, range=(0,255))
        chi, p = chisquare(f_obs=hs + 1, f_exp=ho + 1)
        chi_dev = chi / (image_bins - 1)

    else:
        # audio: quantize to 8-bit for entropy/χ²
        qo = quantize_audio_8bit(original)
        qs = quantize_audio_8bit(suspect)

        # “diff %” → use sample match rate
        changed_pixels = int(np.count_nonzero(qo != qs))
        total_pixels   = int(qo.size)
        diff_percent   = (changed_pixels / max(1,total_pixels)) * 100

        orig_entropy   = shannon_entropy(qo)
        suspect_entropy= shannon_entropy(qs)

        audio_bins = 256
        ho, _ = np.histogram(qo, bins=audio_bins, range=(0,255))
        hs, _ = np.histogram(qs, bins=audio_bins, range=(0,255))
        chi, p = chisquare(f_obs=hs + 1, f_exp=ho + 1)
        chi_dev = chi / (audio_bins - 1)

    # Cards
    create_stat_card(row, "Diff %", f"{diff_percent:.2f}%", f"Changed: {changed_pixels}/{total_pixels}", "orange")
    create_stat_card(row, "Entropy (Orig.)", f"{orig_entropy:.3f} bits", "", "cyan")
    create_stat_card(row, "Entropy (Sus.)", f"{suspect_entropy:.3f} bits", "", "cyan")
    chi_text = "bigger" if chi_dev > 1 else "smaller"
    create_stat_card(row, "Chi-Square Deviation", f"{chi_dev:.2f}x {chi_text} than original", f"p = {p:.3e}", "lightgreen" if p > 0.05 else "red")



# --- to view visual comparison in differece ---
def show_visual_comparison(original, suspect, result_label):
    """Show pixel-wise visual difference as brightness map."""
    diff = np.abs(suspect - original)
    if diff.max() > 0:
        diff_scaled = (diff / diff.max()) * 255
    else:
        diff_scaled = diff

    diff_img = Image.fromarray(diff_scaled.astype(np.uint8))
    img_ctk = ctk.CTkImage(light_image=diff_img, dark_image=diff_img, size=(400, 300))

    result_label.configure(image=img_ctk, text="")
    result_label.image = img_ctk


# --- heatmap of suspicious areas ---
def show_chisquare_heatmap(original, suspect, result_label, block_size=16):
    """
    Splits the image into blocks (e.g. 16×16). In each block, compares the statistical distribution
    of pixel values (via chi-square test) between cover and suspect.
    Highlights blocks where the pixel distribution looks unnaturally different (not just one or two pixels).
    Steganography often disturbs the statistical balance of values (like LSBs becoming too uniform).
    Bright = block is statistically “weird” compared to the original.
    """
    try:
        h, w = original.shape
        heatmap_h = math.ceil(h / block_size)
        heatmap_w = math.ceil(w / block_size)
        heatmap = np.zeros((heatmap_h, heatmap_w))

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # walk through all the steps in chunks, extract 1 block at a time from both suspect and original
                block_orig = original[
                    i : min(i + block_size, h), j : min(j + block_size, w) # use minimum to avoid out of bounds at edges
                ]
                block_sus = suspect[
                    i : min(i + block_size, h), j : min(j + block_size, w)
                ]

                if block_orig.size < 4 or block_sus.size < 4:
                    continue

                orig_hist, _ = np.histogram(block_orig, bins=256, range=(0, 255))
                sus_hist, _ = np.histogram(block_sus, bins=256, range=(0, 255))

                chi, _ = chisquare(f_obs=sus_hist + 1, f_exp=orig_hist + 1) # calculate actual chi square, +1 to avoid division by zero
                # larger chi square value = more suspicious, 
                # smaller chi square value = less suspicious

                # compute safe indices, store chi square score
                idx_i = i // block_size
                idx_j = j // block_size
                if idx_i < heatmap_h and idx_j < heatmap_w:
                    heatmap[idx_i, idx_j] = chi

        # normalize heatmap (0–255) for display
        heatmap_norm = (heatmap / np.max(heatmap)) * 255
        heatmap_img = Image.fromarray(heatmap_norm.astype(np.uint8))

        # display the chisquare heatmap
        img_ctk = ctk.CTkImage(
            light_image=heatmap_img, dark_image=heatmap_img, size=(400, 300)
        )
        result_label.configure(image=img_ctk, text="")
        result_label.image = img_ctk

    except Exception as e:
        print(f"Error in chi-square heatmap: {e}")


def noise_residual_heatmap(suspect, result_label):
    try:
        # Compute noise residual via wavelet denoising
        residual = suspect - denoise_wavelet(suspect, channel_axis=None, mode="soft")

        # Plot residual heatmap
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.imshow(np.abs(residual), cmap="hot")
        ax.axis("off")
        plt.tight_layout(pad=0)

        # Save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        # Convert to CTkImage
        img = Image.open(buf)
        img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(400, 300))

        # Display in result_label
        result_label.configure(image=img_ctk, text="")
        result_label.image = img_ctk

    except Exception as e:
        print(f"Error in noise_residual_heatmap: {e}")


# = = = AUDIO COMPARISON = = =
def normalize_audio(audio):
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


# --- show waveform difference for audio ---
def show_waveform(original, suspect, result_label):
    fig, ax = plt.subplots(2, 1, figsize=(6, 3), dpi=100, sharex=True)

    ax[0].plot(original, color="blue", linewidth=0.5)
    ax[0].set_title("Original Waveform")

    ax[1].plot(suspect, color="red", linewidth=0.5)
    ax[1].set_title("Suspect Waveform")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(500, 250))
    result_label.configure(image=img_ctk, text="")
    result_label.image = img_ctk


# --- show spectrogram for audio ---
def show_spectrogram(original, suspect, result_label, rate=44100):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100)

    ax[0].specgram(original, NFFT=1024, Fs=rate, noverlap=512, cmap="viridis")
    ax[0].set_title("Original Spectrogram")

    ax[1].specgram(suspect, NFFT=1024, Fs=rate, noverlap=512, cmap="viridis")
    ax[1].set_title("Suspect Spectrogram")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(500, 250))
    result_label.configure(image=img_ctk, text="")
    result_label.image = img_ctk


# --- check phase ---
def check_phase_anomalies(audio, samplerate):
    # Compute FFT
    spectrum = fft.fft(audio)
    phase = np.angle(spectrum)

    # Compute phase differences between consecutive samples
    phase_diff = np.diff(phase)

    # A natural signal should have a smooth-ish phase difference distribution
    std_dev = np.std(phase_diff)

    return std_dev


# --- echo presence ---
def check_echo_presence(audio):
    """If suspicious=True, there might be possible hidden data."""
    corr = correlate(audio, audio, mode="full")
    corr = corr[len(corr) // 2 :]  # keep positive lags
    # Look for peaks beyond lag=0
    peaks = np.where(corr > 0.6 * np.max(corr))[0]
    suspicious = [p for p in peaks if p > 100]  # ignore near-zero lag
    return len(suspicious) > 0, suspicious[:5]


# --- noise floor analysis ---
def check_noise_floor(audio):
    """Compare original vs suspect RMS → if suspect noise floor is much higher, mark as suspicious."""
    slice_ = audio[: min(len(audio), 5000)].astype(np.float32)
    slice_ = np.nan_to_num(slice_)  # replace NaN/Inf with 0
    rms = np.sqrt(np.mean(slice_**2)) if slice_.size > 0 else 0
    return rms


# --- show spectrogram for audio ---
def show_spectrogram(original, suspect, result_label, rate=44100):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100)

    ax[0].specgram(original, NFFT=1024, Fs=rate, noverlap=512, cmap="viridis")
    ax[0].set_title("Original Spectrogram")

    ax[1].specgram(suspect, NFFT=1024, Fs=rate, noverlap=512, cmap="viridis")
    ax[1].set_title("Suspect Spectrogram")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(500, 250))
    result_label.configure(image=img_ctk, text="")
    result_label.image = img_ctk


# = = = SUSPICION SCORE = = =
# --- HELPERS ---
def normalize_metric(value, min_val, max_val):
    """Scale value → [0,1] with clipping."""
    if max_val == min_val: 
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))

def quantize_audio_8bit(x):
    """Assumes x is float in [-1,1]; returns uint8 0..255 for entropy/χ²."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0)
    q = np.round(((x + 1.0) * 0.5) * 255).astype(np.uint8)
    return q

def lsb_flip_rate(img_a, img_b):
    """Fraction of pixels whose LSB differs."""
    a = img_a.astype(np.uint8) & 1
    b = img_b.astype(np.uint8) & 1
    return np.mean(a != b)

def residual_energy(arr):
    """Mean absolute residual after wavelet denoise; higher → more high-freq noise."""
    res = arr - denoise_wavelet(arr, channel_axis=None, mode="soft")
    return float(np.mean(np.abs(res)))

def block_chi_heatmap_values(original, suspect, block_size=16):
    """Compute χ² per block (no drawing), return array of χ² values."""
    h, w = original.shape
    H = math.ceil(h / block_size); W = math.ceil(w / block_size)
    heat = np.zeros((H, W), dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            bo = original[i:min(i+block_size, h), j:min(j+block_size, w)]
            bs = suspect[i:min(i+block_size, h), j:min(j+block_size, w)]
            if bo.size < 4 or bs.size < 4: 
                continue
            ho, _ = np.histogram(bo, bins=256, range=(0,255))
            hs, _ = np.histogram(bs, bins=256, range=(0,255))
            chi, _ = chisquare(f_obs=hs + 1, f_exp=ho + 1)
            heat[i//block_size, j//block_size] = chi
    return heat

# --- actual calculations ---
def calculate_image_suspicion(original, suspect):
    # 1) SSIM (1 - SSIM → more suspicious)
    ssim_val = ssim(original, suspect, data_range=255)
    ssim_score = normalize_metric(1.0 - ssim_val, 0.0, 0.4)   # >0.4 diff = strong, 1 - SSIM to add on to suspicion score

    # 2) LSB flip-rate
    lsb_rate = lsb_flip_rate(original, suspect)               # 0..1
    lsb_score = normalize_metric(lsb_rate, 0.0, 0.2)          # >20% flips is big

    # 3) Block χ² (use 90th percentile to focus on hotspots)
    heat = block_chi_heatmap_values(original, suspect, block_size=16)
    if np.max(heat) > 0:
        chi_p90 = np.percentile(heat, 90)                     # only pick values from the 90th percentile
        chi_score = normalize_metric(chi_p90, 0.0, np.max(heat))
    else:
        chi_score = 0.0

    # 4) Residual energy increase
    re_o = residual_energy(original)
    re_s = residual_energy(suspect)
    re_ratio = (re_s / re_o) if re_o > 1e-9 else 1.0
    resid_score = normalize_metric(re_ratio, 1.0, 1.5)        # +50% residual → suspicious

    # Combine (heavier weight on ssim & lsb)
    suspicion = 0.35*ssim_score + 0.35*lsb_score + 0.2*chi_score + 0.10*resid_score
    return float(np.clip(suspicion, 0, 1))


def spectral_flatness(x, nfft=2048):
    """0..1; 1=white-noise-like, 0=tonal. We use rfft power spectrum."""
    x = np.nan_to_num(x).astype(np.float32)
    if x.size == 0: 
        return 0.0
    X = np.fft.rfft(x, n=nfft)
    P = np.abs(X)**2 + 1e-12
    gm = np.exp(np.mean(np.log(P)))
    am = np.mean(P)
    return float(gm / am)

def calculate_audio_suspicion(original, suspect, rate):
    # Quantize for entropy/χ²
    qo = quantize_audio_8bit(original)
    qs = quantize_audio_8bit(suspect)

    # Entropy diff (quantized)
    ent_o = shannon_entropy(qo)
    ent_s = shannon_entropy(qs)
    ent_diff = abs(ent_s - ent_o)
    ent_score = normalize_metric(ent_diff, 0.0, 0.4)

    # χ² on quantized histograms
    ho, _ = np.histogram(qo, bins=256, range=(0,255))
    hs, _ = np.histogram(qs, bins=256, range=(0,255))
    chi, p = chisquare(f_obs=hs + 1, f_exp=ho + 1)
    chi_score = 1 - float(np.clip(p*10, 0, 1))  # small p → suspicious

    # Phase/echo/noise
    phase_std = np.std(np.diff(np.angle(np.fft.fft(suspect))))
    phase_score = normalize_metric(phase_std, 0.0, 1.0)

    echo_flag, _ = check_echo_presence(suspect)
    echo_score = 1.0 if echo_flag else 0.0

    rms = check_noise_floor(suspect)
    rms_score = normalize_metric(rms, 0.0, 0.05)

    # Spectral flatness delta
    sf_o = spectral_flatness(original)
    sf_s = spectral_flatness(suspect)
    sf_score = normalize_metric(sf_s - sf_o, 0.0, 0.15)

    # Combine (heavier χ² + phase/echo; entropy/rms/sf assist)
    suspicion = (
        0.30*chi_score + 0.20*phase_score + 0.20*echo_score +
        0.15*sf_score  + 0.10*ent_score   + 0.05*rms_score
    )
    return float(np.clip(suspicion, 0, 1))

def add_suspicion_card(container, score):
    if score >= 0.6:
        color = "red"
        verdict = "Highly Likely Stego"
    elif score >= 0.3:
        color = "orange"
        verdict = "Suspicious"
    else:
        color = "lightgreen"
        verdict = "Likely Clean"

    create_stat_card(container, "Suspicion Score", f"{score:.2f}", verdict, color)


# = = = PARENT FUNC ACTUAL SHOWING = = =


# --- parent func to call the other comparisons ---
def run_comparison(original_label, suspect_label, bottom_frame):
    original_path = original_label.get_file_path()
    suspect_path = suspect_label.get_file_path()

    # 1. clear old results
    for widget in bottom_frame.winfo_children():
        widget.destroy()

    # 2. compare if the original and suspect path is uploaded
    if not original_path or not suspect_path:
        print("⚠️ Please select both files before comparing.")
        # --- add Compare button back ---
        compare_btn = ctk.CTkButton(
            bottom_frame,
            text="Compare Files",
            fg_color="orange",
            command=lambda: run_comparison(original_label, suspect_label, bottom_frame),
        )
        compare_btn.pack(pady=10)
        return

    # --- add Compare button back ---
    compare_btn = ctk.CTkButton(
        bottom_frame,
        text="Compare Files",
        fg_color="orange",
        command=lambda: run_comparison(original_label, suspect_label, bottom_frame),
    )
    compare_btn.pack(pady=10)

    # 3. ensure that the files uploaded is approved
    ext = os.path.splitext(original_path)[1].lower()
    if ext in IMAGE_EXTS:
        mode = "image"
    elif ext in AUDIO_EXTS:
        mode = "audio"
    else:
        print("⚠️ Unsupported file type")
        return

    # 4. load file based on file type
    if mode == "image":
        try:
            original = np.array(Image.open(original_path).convert("L")) # make the images greyscale
            suspect = np.array(Image.open(suspect_path).convert("L")) # make the images greyscale
        except Exception as e:
            print(f"Error loading images: {e}")
            return
    elif mode == "audio":
        try:
            orig_rate, original = wavfile.read(original_path)
            sus_rate, suspect = wavfile.read(suspect_path)

            # convert stereo → mono if needed
            if original.ndim > 1:
                original = original[:, 0]
            if suspect.ndim > 1:
                suspect = suspect[:, 0]

            # # normalise the audio
            original = normalize_audio(original)
            suspect = normalize_audio(suspect)

        except Exception as e:
            print(f"Error loading audios: {e}")
            return

    # 5. This is the final widget to show final suspicion score
    suspicious_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
    suspicious_container.pack(fill="x", padx=10, pady=10)
    if mode == "image":
        score = calculate_image_suspicion(original, suspect)
        add_suspicion_card(suspicious_container, score)
    else:
        score = calculate_audio_suspicion(original, suspect, sus_rate)
        add_suspicion_card(suspicious_container, score)

    # 6. show all the default widgets
    # --- stats container (row of cards) ---
    stats_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
    stats_container.pack(fill="x", padx=10, pady=10)
    show_stats(original, suspect, stats_container, mode)

    # --- histogram container ---
    # histogram
    hist_frame = ctk.CTkFrame(bottom_frame, corner_radius=10, fg_color="gray25")
    hist_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(
        hist_frame, text="Histogram Comparison", font=("Arial", 14, "bold")
    ).pack(anchor="w", padx=10, pady=(5, 0))
    hist_label = ctk.CTkLabel(hist_frame, text="")
    hist_label.pack(expand=True, fill="both", padx=5, pady=5)
    if mode == "image":
        show_histogram(original, suspect, hist_label, bins=256)
    else:
        show_histogram(original, suspect, hist_label, bins=100)

    # 7. load all of the widgets (based on file type)
    if mode == "image":
        row_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        row_frame.pack(expand=True, fill="both", padx=5, pady=5)

        # configure row_frame as 3 equal columns
        row_frame.grid_columnconfigure((0, 1), weight=1)
        row_frame.grid_rowconfigure((0, 1), weight=1)

        # --- Visual Difference ---
        visual_difference_frame = ctk.CTkFrame(
            row_frame, corner_radius=10, fg_color="gray25"
        )
        visual_difference_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(
            visual_difference_frame,
            text="Visual Difference",
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(5, 0))
        visual_difference_label = ctk.CTkLabel(visual_difference_frame, text="")
        visual_difference_label.pack(expand=True, fill="both", padx=5, pady=5)
        show_visual_comparison(original, suspect, visual_difference_label)

        # --- LSB Map ---
        lsb_frame = ctk.CTkFrame(row_frame, corner_radius=10, fg_color="gray25")
        lsb_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(lsb_frame, text="LSB Map", font=("Arial", 14, "bold")).pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        lsb_label = ctk.CTkLabel(lsb_frame, text="")
        lsb_label.pack(expand=True, fill="both", padx=5, pady=5)
        show_lsb_map(original, suspect, lsb_label)

        # --- chi-square heatmap ---
        heatmap_frame = ctk.CTkFrame(row_frame, corner_radius=10, fg_color="gray25")
        heatmap_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(
            heatmap_frame, text="Chi-Square Heatmap", font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=(5, 0))
        heatmap_label = ctk.CTkLabel(heatmap_frame, text="")
        heatmap_label.pack(expand=True, fill="both", padx=5, pady=5)
        show_chisquare_heatmap(original, suspect, heatmap_label)

        # --- noise residual heatmap ---
        residual_frame = ctk.CTkFrame(row_frame, corner_radius=10, fg_color="gray25")
        residual_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(
            residual_frame, text="Noise Residual Heatmap", font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=(5, 0))
        residual_label = ctk.CTkLabel(residual_frame, text="")
        residual_label.pack(expand=True, fill="both", padx=5, pady=5)
        noise_residual_heatmap(suspect, residual_label)
    else:
        # Stats container
        stats_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        stats_container.pack(fill="x", padx=10, pady=10)

        # --- Phase anomaly ---
        phase_std = check_phase_anomalies(suspect, sus_rate)
        create_stat_card(
            stats_container,
            "Phase Stability",
            f"{phase_std:.3f}",
            "Higher = suspicious",
            "orange",
        )

        # --- Echo check ---
        echo_flag, echo_peaks = check_echo_presence(suspect)
        create_stat_card(
            stats_container,
            "Echo Check",
            "Suspicious" if echo_flag else "Clean",
            f"Peaks: {echo_peaks}",
            "red" if echo_flag else "lightgreen",
        )

        # --- Noise floor ---
        rms = check_noise_floor(suspect)
        create_stat_card(
            stats_container, "Noise Floor RMS", f"{rms:.5f}", "Compare with original"
        )

        # --- Row with audio-specific visualizations ---
        row_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        row_frame.pack(expand=True, fill="both", padx=5, pady=5)
        row_frame.grid_columnconfigure((0, 1), weight=1)

        # --- Waveform ---
        wf_frame = ctk.CTkFrame(row_frame, corner_radius=10, fg_color="gray25")
        wf_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(wf_frame, text="Waveform", font=("Arial", 14, "bold")).pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        wf_label = ctk.CTkLabel(wf_frame, text="")
        wf_label.pack(expand=True, fill="both", padx=5, pady=5)
        show_waveform(original, suspect, wf_label)

        # --- Spectrogram ---
        spec_frame = ctk.CTkFrame(row_frame, corner_radius=10, fg_color="gray25")
        spec_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(spec_frame, text="Spectrogram", font=("Arial", 14, "bold")).pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        spec_label = ctk.CTkLabel(spec_frame, text="")
        spec_label.pack(expand=True, fill="both", padx=5, pady=5)
        show_spectrogram(original, suspect, spec_label, rate=orig_rate)


def create_compare_tab(parent):
    # to update what the person sees
    def update_preview(path, label):
        ext = os.path.splitext(path)[1].lower()
        # base default for images
        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            img = Image.open(path)
            img.thumbnail((250, 250))
            img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))
            label.configure(image=img_ctk, text="")
            label.image = img_ctk

        # for audio waves
        elif ext == ".wav":
            # 1. get audio information (audio text)
            with wave.open(path, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                info_text = (
                    f"Audio file loaded:\n{os.path.basename(path)}\n"
                    f"{wf.getnchannels()} ch, {wf.getframerate()} Hz, {duration:.1f}s"
                )
                label.configure(text=info_text, image=None)

            # 2. load up the waveform preview from scipy
            samplerate, audio_data = wavfile.read(path)
            if audio_data.ndim > 1:  # stereo → use first channel
                audio_data = audio_data[:, 0]

            # 3. plot waveform
            fig, ax = plt.subplots(figsize=(3, 1), dpi=100)
            ax.plot(audio_data, linewidth=0.5, color="dodgerblue")
            ax.set_axis_off()
            plt.tight_layout(pad=0)

            # Save plot to memory
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            # Convert waveform image to CTkImage
            img = Image.open(buf)
            img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 80))

            # Update preview
            label.configure(image=img_ctk, text=info_text, compound="top")
            label.image = img_ctk

        # for others
        else:
            label.configure(
                text=f"Unsupported preview:\n{os.path.basename(path)}", image=None
            )

    # frame = ctk.CTkFrame(parent, fg_color="transparent")
    # frame.pack(expand=True, fill="both")

    scroll_frame = ScrollableFrame(parent, fg_color="gray20")
    scroll_frame.pack(expand=True, fill="both")
    frame = scroll_frame.scrollable_frame  # use this as your main container

    # Configure grid for main frame (2 columns, 2 rows)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_rowconfigure(0, weight=3)  # top takes more space
    frame.grid_rowconfigure(1, weight=3)  # bottom smaller

    # ===== LEFT FRAME =====
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    orig_label = DragDropLabel(left_frame, text="Original File (Drag & Drop / Browse)")
    orig_label.pack(pady=10)

    original_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="gray25")
    original_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(original_frame, text="Original Preview").pack(
        anchor="w", padx=5, pady=(5, 0)
    )
    original_label = ctk.CTkLabel(
        original_frame, text="No file selected", anchor="center"
    )
    original_label.pack(expand=True, fill="both", padx=5, pady=5)

    def on_original_file_selected(path):
        if path and os.path.isfile(path):
            update_preview(path, original_label)

    orig_label.on_file_selected = (
        on_original_file_selected  # indicate that it is selected or not
    )

    # ===== RIGHT FRAME =====
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    stego_label = DragDropLabel(right_frame, text="Suspect File (Drag & Drop / Browse)")
    stego_label.pack(pady=10)

    suspect_frame = ctk.CTkFrame(right_frame, corner_radius=10, fg_color="gray25")
    suspect_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(suspect_frame, text="Suspect Preview").pack(
        anchor="w", padx=5, pady=(5, 0)
    )
    suspect_label = ctk.CTkLabel(
        suspect_frame, text="No file selected", anchor="center"
    )
    suspect_label.pack(expand=True, fill="both", padx=5, pady=5)

    def on_suspect_file_selected(path):
        if path and os.path.isfile(path):
            update_preview(path, suspect_label)

    stego_label.on_file_selected = (
        on_suspect_file_selected  # indicate that it is selected or not
    )

    # ===== BOTTOM FRAME =====
    bottom_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray15")
    bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    compare_btn = ctk.CTkButton(
        bottom_frame,
        text="Compare Files",
        fg_color="orange",
        command=lambda: run_comparison(orig_label, stego_label, bottom_frame),
    )
    compare_btn.pack(pady=10)

    # result_output = ctk.CTkTextbox(bottom_frame, height=200)
    # result_output.pack(padx=5, pady=5, fill="both", expand=True)

    return frame
