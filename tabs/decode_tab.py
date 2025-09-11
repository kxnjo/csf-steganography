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
