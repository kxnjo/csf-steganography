import customtkinter as ctk
from tkinter import messagebox
from assets.dragdrop import DragDropLabel
from encoder_decoder.image_encoder import encode_message as encode_image_message
from encoder_decoder.audio_encoder import file_to_bits, embed_payload, embed_header
from scipy.io import wavfile  # Fixed import
import os
import numpy as np
import tempfile  # For creating temporary files for text payloads

from PIL import Image # to load image preview
import wave
import matplotlib.pyplot as plt
import io
# import simpleaudio as sa

from assets.ScrollableFrame import ScrollableFrame  # custom scrollable frame

# keep global/state reference to current audio
current_play_obj = None

def create_encode_tab(parent):
    # run when a file has been up;paded
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

    # --- frame setup ---
    # frame = ctk.CTkFrame(parent, fg_color="transparent")
    # frame.pack(expand=True, fill="both")

    scroll_frame = ScrollableFrame(parent, fg_color="gray20")
    scroll_frame.pack(expand=True, fill="both")
    frame = scroll_frame.scrollable_frame  # use this as your main container

    # ---------------- Left Panel ----------------
    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    # --- Cover File Section ---
    cover_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="gray25")
    cover_frame.pack(expand=True, fill="x", padx=5, pady=10)
    ctk.CTkLabel(cover_frame, text="Cover File").pack(anchor="w", padx=5, pady=(5, 0))
    cover_label = DragDropLabel(cover_frame, text="Drag & Drop File \n OR \nBrowse File")
    cover_label.pack(padx=5, pady=10, fill="x")
    def on_cover_file_selected(path):
        if path and os.path.isfile(path):
            update_preview(path, preview_label)

    cover_label.on_file_selected = on_cover_file_selected # indicate that it is selected or not

    # --- Secret Data Section ---
    secret_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color="gray25")
    secret_frame.pack(expand=True, fill="x", padx=5, pady=10)
    ctk.CTkLabel(secret_frame, text="Payload (Secret Data)").pack(anchor="w", padx=5, pady=(5, 0))

    tab_var = ctk.StringVar(value="Text Message")
    tab_frame = ctk.CTkFrame(secret_frame, fg_color="transparent")
    tab_frame.pack(fill="x", padx=5, pady=5)
    text_tab_btn = ctk.CTkButton(tab_frame, text="Text Message", width=120, corner_radius=8)
    text_tab_btn.pack(side="left", padx=(0, 5))
    file_tab_btn = ctk.CTkButton(tab_frame, text="File Payload", width=120, corner_radius=8)
    file_tab_btn.pack(side="left")

    content_frame = ctk.CTkFrame(secret_frame, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=5, pady=5)
    msg_entry = ctk.CTkTextbox(content_frame, height=100)
    msg_entry.pack(fill="x")
    file_payload = DragDropLabel(content_frame, text="Drag & Drop File \n OR \nBrowse File")

    def reset_file_payload():
        file_payload.configure(text="Drag & Drop File \n OR \nBrowse File")

    # update the frame if something was uploaded
    def update_tab():
        for widget in content_frame.winfo_children():
            widget.pack_forget()
        if tab_var.get() == "Text Message":
            text_tab_btn.configure(fg_color="dodgerblue", text_color="white")
            file_tab_btn.configure(fg_color="gray30", text_color="gray80")
            msg_entry.configure(state="normal")
            reset_file_payload()
            msg_entry.pack(fill="x")
        else:
            file_tab_btn.configure(fg_color="dodgerblue", text_color="white")
            text_tab_btn.configure(fg_color="gray30", text_color="gray80")
            msg_entry.delete("1.0", "end")
            msg_entry.configure(state="disabled")
            file_payload.pack(fill="x", pady=10)
        
        # update preview window
        cover_path = cover_label.get_file_path()
        # update the side preview window
        if cover_path and os.path.isfile(cover_path):
            update_preview(cover_path, preview_label)

    text_tab_btn.configure(command=lambda: (tab_var.set("Text Message"), update_tab()))
    file_tab_btn.configure(command=lambda: (tab_var.set("File Payload"), update_tab()))
    update_tab()

    # Encryption Key
    ctk.CTkLabel(secret_frame, text="Encryption Key").pack(anchor="w", padx=5)
    key_entry = ctk.CTkEntry(secret_frame, show="*")
    key_entry.pack(padx=5, pady=5, fill="x")
    ctk.CTkLabel(secret_frame, text="This key will be required for decoding the payload",
                  font=ctk.CTkFont(size=10, slant="italic")).pack(anchor="w", padx=5, pady=(0,5))
    

    # --- Bits Selection ---
    ctk.CTkLabel(left_frame, text="Bits per Channel:").pack(anchor="w", padx=5, pady=(5,0))
    bits_option = ctk.CTkOptionMenu(left_frame, values=[f"{i} bit" for i in range(1,9)])
    bits_option.set("1 bit")
    bits_option.pack(padx=5, pady=5, fill="x")

    # --- Start Position ---
    ctk.CTkLabel(left_frame, text="Start Position:").pack(anchor="w", padx=5, pady=(10,0))
    start_entry = ctk.CTkEntry(left_frame)
    start_entry.insert(0, "0")
    start_entry.pack(padx=5, pady=5, fill="x")
    
    # ---------------- Cover Capacity Label ----------------
    capacity_label = ctk.CTkLabel(left_frame, text="Cover Capacity: N/A", anchor="w")
    capacity_label.pack(fill="x", padx=5, pady=(5,0))
    
    # ---------------- Cover Capacity Update ----------------
    from assets.cover_capacity import check_fit 

    def update_cover_capacity():
        cover_path = cover_label.get_file_path()
        try:
            num_lsb = int(bits_option.get().split()[0])

            # Determine payload bytes
            if tab_var.get() == "Text Message":
                payload_bytes = msg_entry.get("1.0", "end").strip().encode("utf-8")
            else:
                payload_path = file_payload.get_file_path()
                if payload_path and os.path.isfile(payload_path):
                    with open(payload_path, "rb") as f:
                        payload_bytes = f.read()
                else:
                    payload_bytes = b""

            result = check_fit(cover_path, num_lsb, payload_bytes)
            max_bits = result["max_bits"]
            payload_bits = result["payload_bits"]
            fit_text = "✅ Fits" if result["fit"] else "❌ Too Large"

            if max_bits is None:
                capacity_label.configure(text="Cover Capacity: Unsupported file")
            else:
                capacity_label.configure(
                    text=f"Cover Capacity: {max_bits} bits | Payload: {payload_bits} bits | {fit_text}"
                )

        except Exception:
            capacity_label.configure(text="Cover Capacity: Error")


    # ---------------- Bindings to update capacity ----------------
    # Update when cover file changes (drag-drop or file dialog)
    cover_label.on_file_selected = lambda path: (update_preview(path, preview_label), update_cover_capacity())

    # Update when LSB selection changes
    bits_option.configure(command=lambda _: update_cover_capacity())

    # Update when payload text changes
    def on_msg_modified(event):
        msg_entry.edit_modified(0)  # reset modified flag
        update_cover_capacity()

    msg_entry.bind("<<Modified>>", on_msg_modified)

    # Update when payload file changes
    file_payload.on_file_selected = lambda path: update_cover_capacity()
    # --- Encode button callback ---
    def run_encode():
        print("PRESSED !! started")
        try:
            cover_path = cover_label.get_file_path()
            key_text = key_entry.get().strip()
            num_lsb = int(bits_option.get().split()[0])
            start_pos = int(start_entry.get())

            if not cover_path or not os.path.isfile(cover_path):
                raise ValueError("Select a valid cover file.")
            if not key_text:
                raise ValueError("Enter an encryption key.")

            # Validate key is integer (required for audio encoding)
            try:
                key_int = int(key_text)
            except ValueError:
                raise ValueError("Encryption key must be a valid integer.")

            cover_ext = os.path.splitext(cover_path)[1].lower()

            if cover_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                # --- Image encoding ---
                if tab_var.get() == "Text Message":
                    message = msg_entry.get("1.0", "end").strip()
                    if not message:
                        raise ValueError("Enter a message to encode.")
                else:
                    payload_path = file_payload.get_file_path()
                    if not payload_path or not os.path.isfile(payload_path):
                        raise ValueError("Select a valid payload file.")
                    with open(payload_path, "rb") as f:
                        file_bytes = f.read()
                    message = file_bytes.decode("latin1")  # keep original bytes

                output_path = encode_image_message(cover_path, message, key_text, num_lsb, start_pos)

            elif cover_ext == ".wav":
                # --- Audio encoding ---
                # For audio, we need to create a payload file regardless of text/file selection
                if tab_var.get() == "Text Message":
                    # Create temporary file with text message
                    message = msg_entry.get("1.0", "end").strip()
                    if not message:
                        raise ValueError("Enter a message to encode.")
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
                    temp_file.write(message)
                    temp_file.close()
                    payload_path = temp_file.name
                    is_temp_file = True
                else:
                    payload_path = file_payload.get_file_path()
                    if not payload_path or not os.path.isfile(payload_path):
                        raise ValueError("Select a valid payload file.")
                    is_temp_file = False

                try:
                    # Read cover audio
                    samplerate, audio_data = wavfile.read(cover_path)
                    if audio_data.dtype != np.int16:
                        audio_data = audio_data.astype(np.int16)
                    audio_data = audio_data.copy()
                    
                    # Read payload bits using your audio_encoder function
                    payload_bits = file_to_bits(payload_path)

                    # Create 32-bit header
                    payload_size = len(payload_bits)
                    header_bits = [(payload_size >> (31 - i)) & 1 for i in range(32)]

                    # Embed header sequentially
                    audio_data = embed_header(audio_data, header_bits, num_lsb)
                    
                    # Calculate how many samples were used for header
                    header_sample_count = (32 + num_lsb - 1) // num_lsb

                    # Embed payload using your audio_encoder function
                    stego_data = embed_payload(audio_data, payload_bits, num_lsb, key_int, offset=header_sample_count)
                    
                    # Save stego audio
                    base_name = os.path.splitext(cover_path)[0]
                    output_path = f"{base_name}_stego.wav"
                    wavfile.write(output_path, samplerate, stego_data)
                    
                finally:
                    # Clean up temporary file if we created one
                    if tab_var.get() == "Text Message" and is_temp_file:
                        os.unlink(payload_path)

            else:
                raise ValueError("Unsupported cover file type. Use images (.png, .jpg, .bmp) or audio (.wav)")
            
            # load up the encoded image into output preview
            update_preview(output_path, output_label)

            messagebox.showinfo("Success", f"✅ Payload encoded!\nSaved at:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    encode_btn = ctk.CTkButton(left_frame, text="Encode Message", fg_color="green", command=run_encode)
    encode_btn.pack(pady=20)

    # ---------------- Right Panel ----------------
    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    # --- Cover Preview Section ---
    preview_frame = ctk.CTkFrame(right_frame, corner_radius=10, fg_color="gray25")
    preview_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(preview_frame, text="Cover Preview").pack(anchor="w", padx=5, pady=(5, 0))
    preview_label = ctk.CTkLabel(preview_frame, text="No file selected", anchor="center")
    preview_label.pack(expand=True, fill="both", padx=5, pady=5)

    # !! functions for preview section at the top

    output_frame = ctk.CTkFrame(right_frame, corner_radius=10, fg_color="gray25")
    output_frame.pack(expand=True, fill="both", padx=5, pady=5)
    ctk.CTkLabel(output_frame, text="Encoded Preview").pack(anchor="w", padx=5, pady=(5, 0))
    output_label = ctk.CTkLabel(output_frame, text="No file selected", anchor="center")
    output_label.pack(expand=True, fill="both", padx=5, pady=5)

    return frame