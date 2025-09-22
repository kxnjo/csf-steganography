import customtkinter as ctk
from dragdrop import DragDropLabel

def create_decode_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    cover_label = DragDropLabel(left_frame, text="Stego File (Drag & Drop / Browse)")
    cover_label.pack(pady=10)

    key_label = ctk.CTkLabel(left_frame, text="Encryption Key")
    key_label.pack(anchor="w", padx=5)
    key_entry = ctk.CTkEntry(left_frame, show="*")
    key_entry.pack(padx=5, pady=5, fill="x")

    decode_btn = ctk.CTkButton(left_frame, text="Decode Message", fg_color="blue")
    decode_btn.pack(pady=20)

    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    msg_label = ctk.CTkLabel(right_frame, text="Decoded Message")
    msg_label.pack(anchor="w", padx=5)
    msg_output = ctk.CTkTextbox(right_frame, height=150)
    msg_output.pack(padx=5, pady=5, fill="both", expand=True)

    return frame
import customtkinter as ctk
from dragdrop import DragDropLabel
import os
from image_decoder import decode_image
from audio_decoder import decode_wav_file

def create_decode_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    # ---------------- Left Panel ----------------
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    cover_label = DragDropLabel(left_frame, text="Stego File (Drag & Drop / Browse)")
    cover_label.pack(pady=10)

    ctk.CTkLabel(left_frame, text="Encryption Key").pack(anchor="w", padx=5)
    key_entry = ctk.CTkEntry(left_frame, show="*")
    key_entry.pack(padx=5, pady=5, fill="x")

    ctk.CTkLabel(left_frame, text="Bits per Channel (for images or audio)").pack(anchor="w", padx=5, pady=(10,0))
    bits_option = ctk.CTkOptionMenu(left_frame, values=[str(i) for i in range(1,9)])
    bits_option.set("1")
    bits_option.pack(padx=5, pady=5, fill="x")

    decode_btn = ctk.CTkButton(left_frame, text="Decode Payload", fg_color="blue")
    decode_btn.pack(pady=20)

    # ---------------- Right Panel ----------------
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    ctk.CTkLabel(right_frame, text="Status / Info").pack(anchor="w", padx=5)
    msg_output = ctk.CTkTextbox(right_frame, height=150)
    msg_output.pack(padx=5, pady=5, fill="both", expand=True)
    msg_output.configure(state="disabled")

    # ---------------- Decode Function ----------------
    def run_decode():
        try:
            stego_path = cover_label.get_file_path()
            if not stego_path or not os.path.isfile(stego_path):
                raise ValueError("Please select a valid stego file.")

            key_text = key_entry.get().strip()
            if not key_text:
                raise ValueError("Please enter the encryption key.")

            num_lsb = int(bits_option.get())
            ext = os.path.splitext(stego_path)[1].lower()
            payload_bytes = None
            save_name = "decoded_payload"

            # Decode based on cover file type
            if ext == ".wav":
                payload_bytes = decode_wav_file(stego_path, num_lsb, int(key_text))
                # Try to detect WAV header
                if payload_bytes[:4] == b"RIFF":
                    save_name += ".wav"
                else:
                    save_name += ".bin"

            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                payload_bytes, _ = decode_image(stego_path, key_text)
                # Auto-detect common file headers
                if payload_bytes.startswith(b"\x89PNG"):
                    save_name += ".png"
                elif payload_bytes.startswith(b"\xFF\xD8"):
                    save_name += ".jpg"
                elif payload_bytes[:4] == b"RIFF":
                    save_name += ".wav"
                elif payload_bytes.startswith(b"%PDF"):
                    save_name += ".pdf"
                else:
                    save_name += ".bin"

            else:
                raise ValueError("Unsupported file type. Use image or WAV.")

            if payload_bytes is None or len(payload_bytes) == 0:
                raise ValueError("Decoded payload is empty.")

            # Save the payload bytes
            save_path = os.path.join(os.getcwd(), save_name)
            with open(save_path, "wb") as f:
                f.write(payload_bytes)

            msg_output.configure(state="normal")
            msg_output.delete("1.0", "end")
            msg_output.insert("1.0", f"Payload decoded successfully:\n{save_path}")
            msg_output.configure(state="disabled")

        except Exception as e:
            msg_output.configure(state="normal")
            msg_output.delete("1.0", "end")
            msg_output.insert("1.0", f"❌ Error: {str(e)}")
            msg_output.configure(state="disabled")

    decode_btn.configure(command=run_decode)

    return frame
