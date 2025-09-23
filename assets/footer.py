import customtkinter as ctk

def create_footer(parent):
    """Creates and returns the footer frame."""
    frame = ctk.CTkFrame(parent, corner_radius=0, fg_color="gray15")
    frame.configure(height=30)

    footer_label = ctk.CTkLabel(
        frame,
        text="LSB Steganography Tool - Secure message hiding using Least Significant Bit encoding",
        font=ctk.CTkFont(size=11),
    )
    footer_label.pack(pady=5)

    return frame
