import customtkinter as ctk
from dragdrop import DragDropLabel
from decode_audio.audio_decode import extract_payload
from tkinter import messagebox  
import chardet

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

    def on_decode():
        filepath = cover_label.filepath
        key = key_entry.get()

        if not filepath or not key:
            messagebox.showerror("Error", "Select a file and enter a key")
            return

        try:
            data = extract_payload(filepath, key)

            try:
                # Try decoding as UTF-8 text
                message_str = data.decode('utf-8')
            except UnicodeDecodeError:
                # If it fails, save as binary
                with open("output.bin", "wb") as f:
                    f.write(data)
                message_str = "<Binary data saved to output.bin>"

            msg_output.delete("1.0", "end")
            msg_output.insert("end", message_str)

        except Exception as e:
            messagebox.showerror("Error", str(e))


    decode_btn.configure(command=on_decode)

    return frame


