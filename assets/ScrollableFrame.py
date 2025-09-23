import customtkinter as ctk

class ScrollableFrame(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        canvas = ctk.CTkCanvas(
            self, 
            bg=self.cget("fg_color"),   # match background color
            highlightthickness=0, 
            bd=0
        )
        scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Keep reference to window_id so we can resize
        window_id = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Stretch content width when resizing
        def _resize_scroll_frame(event):
            canvas.itemconfig(window_id, width=event.width)
        self.bind("<Configure>", _resize_scroll_frame)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ---- Mousewheel bindings ----
        def _on_mousewheel(event):
            # On Windows / Linux
            canvas.yview_scroll(-1 * (event.delta // 120), "units")

        def _on_mousewheel_mac(event):
            # On macOS (event.delta is small)
            canvas.yview_scroll(-1 * int(event.delta), "units")

        # Bind both so it works cross-platform
        canvas.bind_all("<MouseWheel>", _on_mousewheel)      # Windows / Linux
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down
        canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel_mac)  # macOS