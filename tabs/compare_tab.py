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

from scipy.stats import chisquare
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_wavelet
import math

# detect if it is an image (or exceptable file forms)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")

# to display the image
def display_image(frame, path):
    img = Image.open(path)
    img = img.resize((300, 300))  # resize to fit
    tk_img = ImageTk.PhotoImage(img)

    label = ctk.CTkLabel(frame, image=tk_img, text="")
    label.image = tk_img  # keep reference!
    label.pack(padx=10, pady=10)
    return label


def show_histogram(original, suspect, result_label):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=100)

    ax[0].hist(original.ravel(), bins=256, color="blue", alpha=0.7)
    ax[0].set_title("Original Histogram")

    ax[1].hist(suspect.ravel(), bins=256, color="red", alpha=0.7)
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
    value_label = ctk.CTkLabel(card, text=value, font=("Arial", 24, "bold"), text_color=color)
    value_label.pack(anchor="w", padx=10, pady=(5, 0))

    # Subtitle
    if subtitle:
        subtitle_label = ctk.CTkLabel(card, text=subtitle, font=("Arial", 12), text_color="lightgray")
        subtitle_label.pack(anchor="w", padx=10, pady=(0, 10))

    return card
def show_stats(original, suspect, container):
     # pixel difference %
    diff = np.abs(suspect - original)
    changed_pixels = np.count_nonzero(diff)
    total_pixels = diff.size
    diff_percent = (changed_pixels / total_pixels) * 100

    # entropy
    orig_entropy = shannon_entropy(original)
    suspect_entropy = shannon_entropy(suspect)

    # chi-square on histograms
    orig_hist, _ = np.histogram(original, bins=256, range=(0, 255))
    suspect_hist, _ = np.histogram(suspect, bins=256, range=(0, 255))
    chi, p = chisquare(f_obs=suspect_hist + 1, f_exp=orig_hist + 1)

    # Clear old widgets
    for widget in container.winfo_children():
        widget.destroy()

    # Pack stat cards horizontally
    row = ctk.CTkFrame(container, fg_color="transparent")
    row.pack(fill="x", pady=5)

    create_stat_card(
        row, "Pixel Difference", f"{diff_percent:.2f}%",
        f"Changed pixels vs total: {changed_pixels}/{total_pixels}",
        color="orange"
    )

    create_stat_card(
        row, "Entropy (Original)", f"{orig_entropy:.3f} bits",
        "Information content of cover", color="cyan"
    )

    create_stat_card(
        row, "Entropy (Suspect)", f"{suspect_entropy:.3f} bits",
        "Information content of stego", color="cyan"
    )

    create_stat_card(
        row, "Chi-Square", f"{chi:.2f}",
        f"p = {p:.3e}", color="lightgreen" if p > 0.05 else "red"
    )

# --- to view visual comparison in differece ---
def visual_comparison(original_path, suspect_path, frame): 
    """ 
        Directly subtracts the pixel values between original and suspect, 
        normalizes, and shows differences as brightness.
        Every pixel that changed shows up in the map.
        Brighter = bigger pixel difference.
    """
    try:
        original = np.array(Image.open(original_path).convert('L'))
        suspect = np.array(Image.open(suspect_path).convert('L'))
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # compute difference
    diff = np.abs(suspect - original)
    diff_scaled = (diff / diff.max()) * 255  # normalize to 0–255

    # convert numpy array → PIL image
    diff_img = Image.fromarray(diff_scaled.astype(np.uint8))

    # resize for GUI (adjust size as needed)
    img_ctk = ctk.CTkImage(light_image=diff_img, dark_image=diff_img, size=(400, 300))

    # update result label
    visual_difference_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray25")
    visual_difference_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(visual_difference_frame, text="Visual Difference Preview").pack(anchor="w", padx=5, pady=(5, 0))
    visual_difference_label = ctk.CTkLabel(visual_difference_frame, text="", anchor="center")
    visual_difference_label.pack(expand=True, fill="both", padx=5, pady=5)

    # load the image
    visual_difference_label.configure(image=img_ctk, text="")  
    visual_difference_label.image = img_ctk  # keep reference so it doesn’t get GC’d

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
                block_orig = original[i:min(i+block_size, h), j:min(j+block_size, w)]
                block_sus = suspect[i:min(i+block_size, h), j:min(j+block_size, w)]

                if block_orig.size < 4 or block_sus.size < 4:
                    continue

                orig_hist, _ = np.histogram(block_orig, bins=256, range=(0, 255))
                sus_hist, _ = np.histogram(block_sus, bins=256, range=(0, 255))

                chi, _ = chisquare(f_obs=sus_hist + 1, f_exp=orig_hist + 1)

                # compute safe indices
                idx_i = i // block_size
                idx_j = j // block_size
                if idx_i < heatmap_h and idx_j < heatmap_w:
                    heatmap[idx_i, idx_j] = chi

        # normalize heatmap (0–255)
        heatmap_norm = (heatmap / np.max(heatmap)) * 255
        heatmap_img = Image.fromarray(heatmap_norm.astype(np.uint8))

        img_ctk = ctk.CTkImage(light_image=heatmap_img, dark_image=heatmap_img, size=(400, 300))
        result_label.configure(image=img_ctk, text="Chi-Square Heatmap")
        result_label.image = img_ctk

    except Exception as e:
        print(f"Error in chi-square heatmap: {e}")


def noise_residual_heatmap(suspect):
    residual = suspect - denoise_wavelet(suspect, channel_axis=None, mode="soft")
    plt.imshow(np.abs(residual), cmap="hot")

# --- parent func to call the other comparisons ---
def run_comparison(original_path, suspect_path, bottom_frame):
    if not original_path or not suspect_path:
        print("⚠️ Please select both files before comparing.")
        return
    try:
        original = np.array(Image.open(original_path).convert('L'))
        suspect = np.array(Image.open(suspect_path).convert('L'))
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # clear old results
    for widget in bottom_frame.winfo_children():
        widget.destroy()

    # --- add Compare button back ---
    compare_btn = ctk.CTkButton(
        bottom_frame,
        text="Compare Files",
        fg_color="orange",
        command=lambda: run_comparison(original_path, suspect_path, bottom_frame)
    )
    compare_btn.pack(pady=10)

    # --- stats container (row of cards) ---
    stats_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
    stats_container.pack(fill="x", padx=10, pady=10)
    show_stats(original, suspect, stats_container)

    # --- histogram container ---
    hist_frame = ctk.CTkFrame(bottom_frame, corner_radius=10, fg_color="gray25")
    hist_frame.pack(expand=True, fill="both", padx=5, pady=5)
    hist_label = ctk.CTkLabel(hist_frame, text="")
    hist_label.pack(expand=True, fill="both", padx=5, pady=5)
    show_histogram(original, suspect, hist_label)

    # --- lsb container ---
    lsb_frame = ctk.CTkFrame(bottom_frame, corner_radius=10, fg_color="gray25")
    lsb_frame.pack(expand=True, fill="both", padx=5, pady=5)
    lsb_label = ctk.CTkLabel(lsb_frame, text="")
    lsb_label.pack(expand=True, fill="both", padx=5, pady=5)
    show_lsb_map(original, suspect, lsb_label)

    # --- chi-square heatmap container ---
    heatmap_frame = ctk.CTkFrame(bottom_frame, corner_radius=10, fg_color="gray25")
    heatmap_frame.pack(expand=True, fill="both", padx=5, pady=5)
    heatmap_label = ctk.CTkLabel(heatmap_frame, text="")
    heatmap_label.pack(expand=True, fill="both", padx=5, pady=5)
    show_chisquare_heatmap(original, suspect, heatmap_label)

    # --- noise residual heatmap ---
    noise_residual_heatmap


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
                info_text = (f"Audio file loaded:\n{os.path.basename(path)}\n"
                         f"{wf.getnchannels()} ch, {wf.getframerate()} Hz, {duration:.1f}s")
                label.configure(
                    text=info_text,
                    image=None
                )
            
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
                text=f"Unsupported preview:\n{os.path.basename(path)}",
                image=None
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
    ctk.CTkLabel(original_frame, text="Original Preview").pack(anchor="w", padx=5, pady=(5, 0))
    original_label = ctk.CTkLabel(original_frame, text="No file selected", anchor="center")
    original_label.pack(expand=True, fill="both", padx=5, pady=5)
    def on_original_file_selected(path):
        if path and os.path.isfile(path):
            update_preview(path, original_label)

    orig_label.on_file_selected = on_original_file_selected # indicate that it is selected or not

    # ===== RIGHT FRAME =====
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    stego_label = DragDropLabel(right_frame, text="Suspect File (Drag & Drop / Browse)")
    stego_label.pack(pady=10)

    suspect_frame = ctk.CTkFrame(right_frame, corner_radius=10, fg_color="gray25")
    suspect_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(suspect_frame, text="Suspect Preview").pack(anchor="w", padx=5, pady=(5, 0))
    suspect_label = ctk.CTkLabel(suspect_frame, text="No file selected", anchor="center")
    suspect_label.pack(expand=True, fill="both", padx=5, pady=5)
    def on_suspect_file_selected(path):
        if path and os.path.isfile(path):
            update_preview(path, suspect_label)

    stego_label.on_file_selected = on_suspect_file_selected # indicate that it is selected or not


    # ===== BOTTOM FRAME =====
    bottom_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray15")
    bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    compare_btn = ctk.CTkButton(bottom_frame, text="Compare Files", fg_color="orange", command=lambda: run_comparison(orig_label.get_file_path(), stego_label.get_file_path(), bottom_frame))
    compare_btn.pack(pady=10)

    # result_output = ctk.CTkTextbox(bottom_frame, height=200)
    # result_output.pack(padx=5, pady=5, fill="both", expand=True)



    return frame
