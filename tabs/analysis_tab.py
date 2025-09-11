import customtkinter as ctk
from dragdrop import DragDropLabel

def create_analysis_tab(parent):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(expand=True, fill="both")

    top_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="gray20")
    top_frame.pack(expand=True, fill="both", padx=10, pady=10)

    file_label = DragDropLabel(top_frame, text="File for Steganalysis (Drag & Drop / Browse)")
    file_label.pack(pady=10)

    analyze_btn = ctk.CTkButton(top_frame, text="Run Steganalysis", fg_color="purple")
    analyze_btn.pack(pady=20)

    result_label = ctk.CTkLabel(top_frame, text="Steganalysis Results")
    result_label.pack(anchor="w", padx=5)
    result_output = ctk.CTkTextbox(top_frame, height=200)
    result_output.pack(padx=5, pady=5, fill="both", expand=True)

    return frame
