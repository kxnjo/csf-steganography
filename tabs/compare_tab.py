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


def create_compare_tab(parent):
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

    def run_comparison(original_path, suspect_path):
        print("button clicked")
        if not original_path or not suspect_path:
            print("⚠️ Please select both files before comparing.")
            return
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
        result_label.configure(image=img_ctk, text="")  
        result_label.image = img_ctk  # keep reference so it doesn’t get GC’d

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    # scroll_frame = ScrollableFrame(parent, fg_color="gray20")
    # scroll_frame.pack(expand=True, fill="both")
    # frame = scroll_frame.scrollable_frame  # use this as your main container

    # Configure grid for main frame (2 columns, 2 rows)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_rowconfigure(0, weight=3)  # top takes more space
    frame.grid_rowconfigure(1, weight=1)  # bottom smaller

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

    compare_btn = ctk.CTkButton(bottom_frame, text="Compare Files", fg_color="orange", command=lambda: run_comparison(orig_label.get_file_path(), stego_label.get_file_path()))
    compare_btn.pack(pady=10)

    # result_output = ctk.CTkTextbox(bottom_frame, height=200)
    # result_output.pack(padx=5, pady=5, fill="both", expand=True)

    result_frame = ctk.CTkFrame(bottom_frame, corner_radius=10, fg_color="gray25")
    result_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(result_frame, text="Comparison Preview").pack(anchor="w", padx=5, pady=(5, 0))
    result_label = ctk.CTkLabel(result_frame, text="", anchor="center")
    result_label.pack(expand=True, fill="both", padx=5, pady=5)


    return frame
