import customtkinter as ctk
from dragdrop import DragDropLabel
from tkinter import filedialog, messagebox
import encoding

def create_encode_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    # Left panel
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    cover_label = DragDropLabel(left_frame, text="Drag & Drop File \n OR \nBrowse File")
    cover_label.pack(pady=10)
    
    # Initialize file path storage
    cover_label.file_path = None
    
    # Set up file path storage for the cover label
    def on_file_drop(files):
        if files:
            cover_label.file_path = files[0]
            print(f"File selected: {files[0]}")  # Debug print
        else:
            cover_label.file_path = None
            print("No file selected")  # Debug print
    
    cover_label.on_drop = on_file_drop
    
    msg_label = ctk.CTkLabel(left_frame, text="Secret Message")
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

    # Store references for the encoding function
    encode_btn.configure(command=lambda: handle_encode_click(cover_label, msg_entry, key_entry, bits_option, start_entry))

    return frame

def handle_encode_click(cover_label, msg_entry, key_entry, bits_option, start_entry):
    """Collect UI inputs and call core encoding function."""
    try:
        cover_path = cover_label.file_path
        message = msg_entry.get("1.0", "end-1c").strip()
        key_text = key_entry.get().strip()
        num_lsb = int(bits_option.get().split()[0])
        start_pos = int(start_entry.get())

        output_path = encoding.encode_message(
            cover_path=cover_path,
            message=message,
            key_text=key_text,
            num_lsb=num_lsb,
            start_pos=start_pos,
        )

        messagebox.showinfo("Success", f"Message encoded successfully!\nSaved as: {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Encoding failed: {str(e)}")
