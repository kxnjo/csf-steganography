import customtkinter as ctk
from dragdrop import DragDropLabel

def create_encode_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    # Left panel
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    cover_label = DragDropLabel(left_frame, text="Upload Cover File")
    cover_label.pack(pady=10)

    payload_label = DragDropLabel(left_frame, text="Upload Payload File")
    payload_label.pack(pady=10)
    
    msg_label = ctk.CTkLabel(left_frame, text="Or Manually Enter Secret Message")
    msg_label.pack(anchor="w", padx=5)
    msg_entry = ctk.CTkTextbox(left_frame, height=100)
    msg_entry.pack(padx=5, pady=5, fill="x")

    key_label = ctk.CTkLabel(left_frame, text="Encryption Key")
    key_label.pack(anchor="w", padx=5)
    key_entry = ctk.CTkEntry(left_frame, show="*")
    key_entry.pack(padx=5, pady=5, fill="x")

    # Right panel
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    bits_label = ctk.CTkLabel(right_frame, text="Bits per Channel:")
    bits_label.pack(anchor="w", padx=5, pady=(5, 0))
    bits_option = ctk.CTkOptionMenu(right_frame, values=["1 bit", "2 bits", "3 bits", "4 bits", "5 bits", "6 bits", "7 bits", "8 bits"])
    bits_option.pack(padx=5, pady=5, fill="x")

    start_label = ctk.CTkLabel(right_frame, text="Start Position:")
    start_label.pack(anchor="w", padx=5, pady=(10, 0))
    start_entry = ctk.CTkEntry(right_frame)
    start_entry.insert(0, "0")
    start_entry.pack(padx=5, pady=5, fill="x")

    encode_btn = ctk.CTkButton(right_frame, text="Encode Message", fg_color="green")
    encode_btn.pack(pady=20)

    return frame
