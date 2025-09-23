import customtkinter as ctk
from dragdrop import DragDropLabel
from PIL import Image, ImageTk

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
    def run_comparison():
        cover_path = getattr(orig_label, "file_path", None)
        stego_path = getattr(stego_label, "file_path", None)

        if not cover_path or not stego_path:
            result_label.configure(text="Please select both files")
            return

        # Check if images
        if cover_path.lower().endswith(IMAGE_EXTS) and stego_path.lower().endswith(IMAGE_EXTS):
            # Clear results panel
            for widget in right_frame.winfo_children():
                widget.destroy()

            # Show images side by side
            img_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
            img_frame.pack(expand=True, fill="both")

            left_img = display_image(img_frame, cover_path)
            left_img.pack(side="left", expand=True, padx=10, pady=10)

            right_img = display_image(img_frame, stego_path)
            right_img.pack(side="right", expand=True, padx=10, pady=10)

            return

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    orig_label = DragDropLabel(left_frame, text="Cover File (Drag & Drop / Browse)")
    orig_label.pack(pady=10)

    stego_label = DragDropLabel(left_frame, text="Stego File (Drag & Drop / Browse)")
    stego_label.pack(pady=10)

    compare_btn = ctk.CTkButton(left_frame, text="Compare Files", fg_color="orange", )
    compare_btn.pack(pady=20)

    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    result_label = ctk.CTkLabel(right_frame, text="Comparison Results")
    result_label.pack(anchor="w", padx=5)
    result_output = ctk.CTkTextbox(right_frame, height=200)
    result_output.pack(padx=5, pady=5, fill="both", expand=True)

    return frame
