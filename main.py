import customtkinter as ctk
from tkinterdnd2 import TkinterDnD

class CTkDnD(ctk.CTk, TkinterDnD.Tk):
    """A root window that combines CustomTkinter and TkinterDnD2."""
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        TkinterDnD.Tk.__init__(self, *args, **kwargs)

def main():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = CTkDnD()
    app.title("LSB Steganography Tool")
    app.geometry("900x720")
    
    # --- Safe close handler ---
    def on_closing():
        # exit mainloop first, then destroy
        app.quit()
        app.after(100, app.destroy)

    app.protocol("WM_DELETE_WINDOW", on_closing)

    # import and build your UI
    from header import create_header
    from footer import create_footer
    from tabs.encode_tab import create_encode_tab
    from tabs.decode_tab import create_decode_tab
    from tabs.compare_tab import create_compare_tab
    from tabs.analysis_tab import create_analysis_tab

    # build layout (like you did before)
    header = create_header(app)
    header.pack(fill="x")

    notebook = ctk.CTkTabview(app, fg_color="gray15", corner_radius=15)
    notebook.pack(expand=True, fill="both", padx=0, pady=0)

    encode_tab = notebook.add("Encode")
    decode_tab = notebook.add("Decode")
    compare_tab = notebook.add("Compare")
    analysis_tab = notebook.add("Steganalysis")

    create_encode_tab(encode_tab)
    create_decode_tab(decode_tab)
    create_compare_tab(compare_tab)
    create_analysis_tab(analysis_tab)

    footer = create_footer(app)
    footer.pack(fill="x")


    app.mainloop()

if __name__ == "__main__":
    main()
