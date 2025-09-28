import customtkinter as ctk
from assets.dragdrop import DragDropLabel
import os
from tkinter import messagebox, filedialog
from encoder_decoder.image_decoder import decode_image
from encoder_decoder.audio_decode import decode_wav_file_gui
from encoder_decoder.video import extract_audio
import numpy as np
from scipy.io import wavfile
import random
import mimetypes

from PIL import Image # to load image preview
import wave
import matplotlib.pyplot as plt
import io

from assets.ScrollableFrame import ScrollableFrame  # custom scrollable frame

# Improved Audio decoder functions
def extract_payload_with_termination(stego_audio, num_lsb, key):
    """Extract payload bits, stopping when we find consecutive null bytes"""
    num_samples = len(stego_audio)
    
    # Generate the same shuffled indices as the encoder
    indices = list(range(num_samples))
    random.seed(key)
    random.shuffle(indices)
    
    # Extract bits until we find termination pattern
    bits = []
    consecutive_zeros = 0
    max_consecutive_zeros = 16  # Stop after 16 consecutive zero bits (2 null bytes)
    
    for sample_idx in indices:
        sample = stego_audio[sample_idx]
        for l in range(num_lsb):
            bit = (sample >> l) & 1
            bits.append(bit)
            
            # Check for termination (consecutive zeros)
            if bit == 0:
                consecutive_zeros += 1
                if consecutive_zeros >= max_consecutive_zeros:
                    return bits[:-max_consecutive_zeros]  # Remove the termination sequence
            else:
                consecutive_zeros = 0
    
    return bits

def detect_file_type(data):
    """Detect file type from magic bytes"""
    if len(data) < 4:
        return "text", ".txt"
    
    # Common file signatures
    magic_bytes = {
        b"\x89PNG": ("image", ".png"),
        b"\xFF\xD8": ("image", ".jpg"),
        b"\x47\x49\x46": ("image", ".gif"),
        b"BM": ("image", ".bmp"),
        b"%PDF": ("document", ".pdf"),
        b"PK\x03\x04": ("archive", ".zip"),
        b"Rar!": ("archive", ".rar"),
        b"\x7fELF": ("executable", ".elf"),
        b"MZ": ("executable", ".exe"),
        b"RIFF": ("audio", ".wav"),
        b"ID3": ("audio", ".mp3"),
        b"\x1a\x45\xdf\xa3": ("video", ".mkv"),  # Matroska
        b"\x00\x00\x00 ftyp": ("video", ".mp4"),
    }
    
    for magic, (file_type, extension) in magic_bytes.items():
        if data.startswith(magic):
            return file_type, extension
    
    # Check if it's text
    if is_text_data(data):
        return "text", ".txt"
    
    # Default to binary
    return "binary", ".bin"

def is_text_data(data, threshold=0.85):
    """Check if data is likely text"""
    if len(data) == 0:
        return False
    check_data = data[:1000] if len(data) > 1000 else data
    printable_count = sum(1 for byte in check_data if 32 <= byte <= 126 or byte in [9, 10, 13])
    return (printable_count / len(check_data)) > threshold

def decode_wav_file_improved(stego_file, num_lsb, key):
    """Improved WAV decoder that handles both text and files"""
    try:
        samplerate, stego_data = wavfile.read(stego_file)

        # Convert stereo to mono if needed
        if len(stego_data.shape) > 1:
            stego_data = stego_data[:, 0]

        if stego_data.dtype != np.int16:
            stego_data = stego_data.astype(np.int16)

        # Extract all available bits (no early termination for files)
        num_samples = len(stego_data)
        indices = list(range(num_samples))
        random.seed(key)
        random.shuffle(indices)
        
        bits = []
        for sample_idx in indices:
            sample = stego_data[sample_idx]
            for l in range(num_lsb):
                bits.append((sample >> l) & 1)

        # Convert to bytes
        raw_bytes = bytearray()
        for i in range(0, len(bits) - (len(bits) % 8), 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | bits[i + j]
            raw_bytes.append(byte_val)

        return bytes(raw_bytes)
        
    except Exception as e:
        print(f"Audio decoding error: {e}")
        return b''

def create_decode_tab(parent):
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

        elif ext == ".mp4":
            import cv2
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Convert BGR (OpenCV) → RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((250, 250))
                img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))
                label.configure(image=img_ctk, text="")
                label.image = img_ctk
            else:
                label.configure(text=f"Cannot read video frames:\n{os.path.basename(path)}", image=None)


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

    # ---------------- Left Panel ----------------
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    # Stego File Section
    stego_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="gray25")
    stego_frame.pack(fill="x", padx=5, pady=5)
    
    ctk.CTkLabel(stego_frame, text="Stego File").pack(anchor="w", padx=5, pady=(5, 0))
    
    cover_label = DragDropLabel(
        stego_frame, 
        text="Drag & Drop File \n OR \nBrowse File",
        width=300,
        height=120
    )
    cover_label.pack(padx=5, pady=10, fill="x")

    cover_preview_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="gray25")
    cover_preview_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(cover_preview_frame, text="Original Preview").pack(anchor="w", padx=5, pady=(5, 0))
    cover_preview_label = ctk.CTkLabel(cover_preview_frame, text="No file selected", anchor="center")
    cover_preview_label.pack(expand=True, fill="both", padx=5, pady=5)
    def on_original_file_selected(path):
        if path and os.path.isfile(path):
            update_preview(path, cover_preview_label)

    cover_label.on_file_selected = on_original_file_selected # indicate that it is selected or not

    # Decoding Options Section
    options_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="gray25")
    options_frame.pack(fill="x", padx=5, pady=5)
    
    ctk.CTkLabel(options_frame, text="Decoding Options").pack(anchor="w", padx=5, pady=(5, 0))
    
    ctk.CTkLabel(options_frame, text="Encryption Key").pack(anchor="w", padx=5, pady=(10, 0))
    key_entry = ctk.CTkEntry(options_frame, show="*")
    key_entry.pack(padx=5, pady=5, fill="x")

    ctk.CTkLabel(options_frame, text="Bits per Channel").pack(anchor="w", padx=5, pady=(10, 0))
    bits_option = ctk.CTkOptionMenu(options_frame, values=[str(i) for i in range(1, 9)])
    bits_option.set("1")
    bits_option.pack(padx=5, pady=5, fill="x")

    # Auto-save option
    auto_save_var = ctk.BooleanVar(value=True)
    auto_save_check = ctk.CTkCheckBox(options_frame, text="Auto-save detected files", variable=auto_save_var)
    auto_save_check.pack(anchor="w", padx=5, pady=5)

    decode_btn = ctk.CTkButton(left_frame, text="Decode Payload", fg_color="blue")
    decode_btn.pack(pady=20)

    # ---------------- Right Panel ----------------
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    # Results Section
    results_frame = ctk.CTkFrame(right_frame, corner_radius=10, fg_color="gray25")
    results_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    ctk.CTkLabel(results_frame, text="Decoded Output").pack(anchor="w", padx=5, pady=(5, 0))
    
    msg_output = ctk.CTkTextbox(results_frame, height=150)
    msg_output.pack(fill="both", expand=True, padx=5, pady=5)
    msg_output.configure(state="disabled")

    # File info display
    file_info_label = ctk.CTkLabel(results_frame, text="No file decoded yet")
    file_info_label.pack(anchor="w", padx=5, pady=5)

    save_btn = ctk.CTkButton(results_frame, text="Save File As...", fg_color="green")
    save_btn.pack(pady=5)
    save_btn.configure(state="disabled")

    # Open folder button
    open_folder_btn = ctk.CTkButton(results_frame, text="Open Output Folder", fg_color="blue")
    open_folder_btn.pack(pady=5)
    open_folder_btn.configure(state="disabled")

    # Store the current payload data for saving
    current_payload_data = None
    current_file_type = None
    current_extension = None
    current_save_path = None

    # ---------------- Decode Function ----------------
    def run_decode():
        nonlocal current_payload_data, current_file_type, current_extension, current_save_path
        
        try:
            stego_path = cover_label.get_file_path()
            if not stego_path or not os.path.isfile(stego_path):
                raise ValueError("Please select a valid stego file.")

            key_text = key_entry.get().strip()
            if not key_text:
                raise ValueError("Please enter the encryption key.")

            num_lsb = int(bits_option.get())
            ext = os.path.splitext(stego_path)[1].lower()
            
            # Clear previous results
            msg_output.configure(state="normal")
            msg_output.delete("1.0", "end")
            file_info_label.configure(text="Decoding...")
            save_btn.configure(state="disabled")
            open_folder_btn.configure(state="disabled")
            current_payload_data = None
            current_file_type = None
            current_extension = None
            current_save_path = None

            # Decode based on file type
            if ext == ".wav":
                # Validate key for audio (must be integer)
                try:
                    key_int = int(key_text)
                except ValueError:
                    raise ValueError("For audio files, encryption key must be a valid integer.")
                # run decode audio here
                payload_bytes = decode_wav_file_gui(stego_path, num_lsb, key_int)
                
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                payload_bytes, _ = decode_image(stego_path, key_text)
            
            # Inside run_decode(), replacing video decoding part
            elif ext in [".mp4", ".mkv"]:
                key_int = int(key_text)
                audio_file = extract_audio(stego_path)
                payload_bytes = decode_wav_file_gui(audio_file, num_lsb, key_int)

            else:
                raise ValueError("Unsupported file type. Use image (.png, .jpg, .bmp) or audio (.wav) files.")

            if payload_bytes is None or len(payload_bytes) == 0:
                raise ValueError("Decoded payload is empty.")

            current_payload_data = payload_bytes

            # Detect file type
            file_type, extension = detect_file_type(payload_bytes)
            current_file_type = file_type
            current_extension = extension

            # Generate save path
            base_name = "decoded_file"
            save_path = os.path.join(os.getcwd(), f"{base_name}{extension}")
            counter = 1
            while os.path.exists(save_path):
                save_path = os.path.join(os.getcwd(), f"{base_name}_{counter}{extension}")
                counter += 1

            current_save_path = save_path

            # Auto-save if enabled
            if auto_save_var.get():
                with open(save_path, "wb") as f:
                    f.write(payload_bytes)
                file_info_label.configure(text=f"✅ Auto-saved: {os.path.basename(save_path)}")
            else:
                file_info_label.configure(text=f"✅ Ready to save: {os.path.basename(save_path)}")

            # Display results
            msg_output.insert("1.0", f"✅ File decoded successfully!\n\n")
            msg_output.insert("end", f"File type: {file_type} ({extension})\n")
            msg_output.insert("end", f"File size: {len(payload_bytes)} bytes\n")
            msg_output.insert("end", f"Detected as: {get_file_description(file_type, extension)}\n\n")

            if file_type == "text":
                try:
                    text_content = payload_bytes.decode('utf-8')
                    if len(text_content) <= 1000:  # Only display if not too long
                        msg_output.insert("end", f"Text content:\n{text_content}\n")
                    else:
                        msg_output.insert("end", f"Text content (first 500 chars):\n{text_content[:500]}...\n")
                except:
                    msg_output.insert("end", "Binary content (cannot display as text)\n")
            else:
                msg_output.insert("end", f"Binary file - use 'Save File As...' to download\n")

            if not auto_save_var.get():
                save_btn.configure(state="normal")
            open_folder_btn.configure(state="normal")

        except Exception as e:
            msg_output.configure(state="normal")
            msg_output.delete("1.0", "end")
            msg_output.insert("1.0", f"❌ Error: {str(e)}")
            msg_output.configure(state="disabled")
            file_info_label.configure(text="Decoding failed")
            save_btn.configure(state="disabled")
            open_folder_btn.configure(state="disabled")

    def get_file_description(file_type, extension):
        """Get user-friendly file description"""
        descriptions = {
            ".png": "PNG Image",
            ".jpg": "JPEG Image", 
            ".gif": "GIF Image",
            ".bmp": "Bitmap Image",
            ".pdf": "PDF Document",
            ".txt": "Text File",
            ".zip": "ZIP Archive",
            ".rar": "RAR Archive",
            ".exe": "Windows Executable",
            ".wav": "WAV Audio",
            ".mp3": "MP3 Audio",
            ".mp4": "MP4 Video",
            ".mkv": "Matroska Video",
            ".bin": "Binary File"
        }
        return descriptions.get(extension, f"{file_type} file")

    def save_file_as():
        """Save file with custom filename"""
        if current_payload_data is None:
            messagebox.showerror("Error", "No payload data to save.")
            return
            
        # Suggest appropriate file types based on detection
        file_types = []
        if current_extension:
            desc = get_file_description(current_file_type, current_extension)
            file_types.append((f"{desc} (*{current_extension})", f"*{current_extension}"))
        
        file_types.extend([
            ("All files", "*.*"),
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("Document files", "*.pdf *.txt *.doc *.docx"),
            ("Audio files", "*.wav *.mp3 *.ogg *.flac"),
            ("Video files", "*.mp4 *.mkv *.avi *.mov"),
            ("Archive files", "*.zip *.rar *.7z"),
            ("Binary files", "*.bin")
        ])
        
        filename = filedialog.asksaveasfilename(
            title="Save decoded file as...",
            defaultextension=current_extension,
            filetypes=file_types
        )
        
        if filename:
            try:
                with open(filename, "wb") as f:
                    f.write(current_payload_data)
                messagebox.showinfo("Success", f"File saved as:\n{filename}")
                file_info_label.configure(text=f"Saved as: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def open_output_folder():
        """Open the folder containing the decoded file"""
        if current_save_path and os.path.exists(current_save_path):
            folder_path = os.path.dirname(current_save_path)
            os.startfile(folder_path)
        else:
            folder_path = os.getcwd()
            os.startfile(folder_path)

    # Connect the buttons
    decode_btn.configure(command=run_decode)
    save_btn.configure(command=save_file_as)
    open_folder_btn.configure(command=open_output_folder)

    return frame
