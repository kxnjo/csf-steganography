import customtkinter as ctk

def create_header(parent):
    frame = ctk.CTkFrame(parent, corner_radius=0)
    frame.configure(height=60)

    title = ctk.CTkLabel(
        frame, 
        text="LSB Steganography Tool",
        font=ctk.CTkFont(size=20, weight="bold")
    )
    title.pack(pady=10)

    subtitle = ctk.CTkLabel(
        frame,
        text="Hide and extract secret messages in images and audio files using Least Significant Bit encoding",
        font=ctk.CTkFont(size=12)
    )
    subtitle.pack()

    return frame 
