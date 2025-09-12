import customtkinter as ctk
from dragdrop import DragDropLabel
import os
import random
from PIL import Image
import numpy as np
from tkinter import filedialog, messagebox

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
    encode_btn.configure(command=lambda: encode_message(cover_label, msg_entry, key_entry, bits_option, start_entry))

    return frame

def get_key_permutation(key, num_lsb):
    """Generate a permutation of LSB positions based on the key."""
    random.seed(key)
    positions = list(range(num_lsb))
    random.shuffle(positions)
    return positions

def text_to_binary(text):
    """Convert text to binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_bytes(binary_string):
    """Convert binary string to bytes."""
    # Pad the binary string to be divisible by 8
    while len(binary_string) % 8 != 0:
        binary_string += '0'
    
    # Convert to bytes
    byte_array = bytearray()
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        byte_array.append(int(byte, 2))
    
    return bytes(byte_array)

def embed_data_in_image(image_array, data_bits, num_lsb, start_pos, key):
    """Embed binary data into image using LSB steganography with 56-bit metadata header.
    - start_pos is stored in the header as a pixel index (not channels).
    - Payload embedding converts start_pos → channel index internally.
    """
    height, width, channels = image_array.shape
    total_channels = height * width * channels

    # Capacity check (account for 56 header bits)
    max_bits = (total_channels - 56) * num_lsb
    if len(data_bits) > max_bits:
        raise ValueError(f"Message too long. Maximum {max_bits} bits can be embedded.")

    flat_image = image_array.flatten()

    # --- Metadata header (56 bits) ---
    length_bits = format(len(data_bits), '032b')   # message length in bits
    num_lsb_bits = format(num_lsb, '08b')          # how many LSBs used
    start_pos_bits = format(start_pos, '016b')     # start position in PIXELS
    header_bits = length_bits + num_lsb_bits + start_pos_bits

    # Embed header sequentially in first 56 flat positions, LSB0 only
    for i, bit in enumerate(header_bits):
        pixel_value = int(flat_image[i])
        pixel_value = (pixel_value & ~1) | int(bit)
        flat_image[i] = np.uint8(pixel_value)

    # --- Payload embedding ---
    lsb_positions = get_key_permutation(key, num_lsb)

    # Ensure payload starts **after header**
    current_pos = max(start_pos * channels, 56)

    bit_index = 0
    for bit in data_bits:
        if current_pos >= len(flat_image):
            break

        pixel_value = int(flat_image[current_pos])
        lsb_index = bit_index % num_lsb
        target_lsb = lsb_positions[lsb_index]

        pixel_value = (pixel_value & ~(1 << target_lsb)) | (int(bit) << target_lsb)
        flat_image[current_pos] = np.uint8(pixel_value)

        bit_index += 1
        if bit_index % num_lsb == 0:
            current_pos += 1

    return flat_image.reshape(height, width, channels)




def encode_message(cover_label, msg_entry, key_entry, bits_option, start_entry):
    """Main encoding function."""
    try:
        # Get input values
        cover_path = cover_label.file_path
        message = msg_entry.get("1.0", "end-1c").strip()
        key_text = key_entry.get().strip()
        num_lsb = int(bits_option.get().split()[0])
        start_pos = int(start_entry.get())
        
        # Debug information
        print(f"Cover path: {cover_path}")
        print(f"Message: {message}")
        print(f"Key: {key_text}")
        print(f"LSB count: {num_lsb}")
        print(f"Start position: {start_pos}")
        
        # Validate inputs
        if not cover_path:
            messagebox.showerror("Error", "Please select a cover image file.")
            return
        
        if not message:
            messagebox.showerror("Error", "Please enter a message to encode.")
            return
        
        if not key_text:
            messagebox.showerror("Error", "Please enter an encryption key.")
            return
        
        if not os.path.exists(cover_path):
            messagebox.showerror("Error", "Cover image file not found.")
            return
        
        # Convert key to integer
        try:
            key = int(key_text)
        except ValueError:
            messagebox.showerror("Error", "Encryption key must be a valid integer.")
            return
        
        # Load and process the image
        image = Image.open(cover_path)
        
        # Convert to RGB if necessary (handle different formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for manipulation
        image_array = np.array(image)
        
        # Convert message to binary
        message_binary = text_to_binary(message)
        
        # Embed the data
        stego_array = embed_data_in_image(image_array, message_binary, num_lsb, start_pos, key)
        
        # Convert back to PIL Image
        stego_image = Image.fromarray(stego_array.astype(np.uint8))
        
        # Save the stego image
        base_name = os.path.splitext(cover_path)[0]
        output_path = f"{base_name}_stego.png"
        
        stego_image.save(output_path, "PNG")
        
        messagebox.showinfo("Success", f"Message encoded successfully!\nSaved as: {output_path}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Encoding failed: {str(e)}")
