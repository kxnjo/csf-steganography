import customtkinter as ctk
from dragdrop import DragDropLabel
from tkinter import filedialog, messagebox
import decoding  # <-- uses decoding.py shown above


def create_decode_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    # Left panel
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    cover_label = DragDropLabel(left_frame, text="Stego File (Drag & Drop / Browse)")
    cover_label.pack(pady=10)

    # track dropped/browsed file path (mirror encode_tab pattern)
    cover_label.file_path = None

    def on_file_drop(files):
        if files:
            cover_label.file_path = files[0]
            print(f"Stego selected: {files[0]}")
        else:
            cover_label.file_path = None
            print("No stego file selected")

    cover_label.on_drop = on_file_drop

    key_label = ctk.CTkLabel(left_frame, text="Encryption Key")
    key_label.pack(anchor="w", padx=5)
    key_entry = ctk.CTkEntry(left_frame, show="*")
    key_entry.pack(padx=5, pady=5, fill="x")

    decode_btn = ctk.CTkButton(left_frame, text="Decode Message", fg_color="blue")
    decode_btn.pack(pady=20)

    # Right panel
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    msg_label = ctk.CTkLabel(right_frame, text="Decoded Message")
    msg_label.pack(anchor="w", padx=5)
    msg_output = ctk.CTkTextbox(right_frame, height=150)
    msg_output.pack(padx=5, pady=5, fill="both", expand=True)

    meta_label = ctk.CTkLabel(right_frame, text="Metadata")
    meta_label.pack(anchor="w", padx=5, pady=(10, 0))
    meta_output = ctk.CTkTextbox(right_frame, height=100)
    meta_output.pack(padx=5, pady=5, fill="both", expand=True)

    def handle_decode_click():
        try:
            stego_path = cover_label.file_path
            key_text = key_entry.get().strip()
            if not stego_path:
                raise ValueError("Please select a stego image file.")
            if not key_text:
                raise ValueError("Please enter the decoding key.")

            # Try text decode first (convenience for message workflow)
            text, meta = decoding.decode_image_to_text(stego_path, key_text)

            # Write text to UI
            msg_output.delete("1.0", "end")
            msg_output.insert("1.0", text)

            # Show metadata
            meta_output.delete("1.0", "end")
            for k, v in meta.items():
                meta_output.insert("end", f"{k}: {v}\n")

            messagebox.showinfo("Success", "Decoding completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {e}")

    decode_btn.configure(command=handle_decode_click)

    return frame
