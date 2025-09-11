import customtkinter as ctk
from dragdrop import DragDropLabel

def create_compare_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    left_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    orig_label = DragDropLabel(left_frame, text="Original File (Drag & Drop / Browse)")
    orig_label.pack(pady=10)

    stego_label = DragDropLabel(left_frame, text="Stego File (Drag & Drop / Browse)")
    stego_label.pack(pady=10)

    compare_btn = ctk.CTkButton(left_frame, text="Compare Files", fg_color="orange")
    compare_btn.pack(pady=20)

    right_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    result_label = ctk.CTkLabel(right_frame, text="Comparison Results")
    result_label.pack(anchor="w", padx=5)
    result_output = ctk.CTkTextbox(right_frame, height=200)
    result_output.pack(padx=5, pady=5, fill="both", expand=True)

    return frame
