import customtkinter as ctk
from tkinter import filedialog, Menu
import os

class DragDropLabel(ctk.CTkLabel):
    """
    Drag-and-drop label that also opens file explorer on click.
    Supports text, images, PDFs, executables, audio.
    Right-click clears the label.
    """

    ACCEPTED_EXTENSIONS = (
        ".txt", ".pdf", ".exe", ".png", ".jpg", ".jpeg", ".gif", ".bmp",
        ".mp3", ".wav", ".ogg", ".flac"
    )

    def __init__(self, parent, text="Drag & Drop or Click to Select File", on_drop=None, **kwargs):
        super().__init__(parent, text=text, **kwargs)

        self.default_text = text
        self.on_drop = on_drop
        self.file_path = None  # <-- store the selected file path

        # Fixed size and wrap long filenames
        self.fixed_width = kwargs.get("width", 300)
        self.configure(corner_radius=10, fg_color="gray20", width=self.fixed_width,
                       height=120, justify="center", wraplength=self.fixed_width - 20)

        # Drag-and-drop
        self.drop_target_register("DND_Files")
        self.dnd_bind("<<Drop>>", self._on_drop)
        self.dnd_bind("<<DragLeave>>", self._on_drag_leave)

        # Click to open file dialog
        self.bind("<Button-1>", lambda e: self._open_file_dialog())

        # Right-click menu for clearing
        self.menu = Menu(parent, tearoff=0)
        self.menu.add_command(label="Clear", command=self._clear)
        self.bind("<Button-3>", self._show_menu)

    def _on_drop(self, event):
        files_or_text = self._parse_drop(event.data)
        dropped_files = [f for f in files_or_text if os.path.isfile(f)]
        dropped_texts = [f for f in files_or_text if not os.path.isfile(f)]

        filtered_files = [
            f for f in dropped_files if f.lower().endswith(self.ACCEPTED_EXTENSIONS)
        ]

        if filtered_files:
            self.file_path = filtered_files[0]  # save first valid file
            display_items = [self._shorten_name(os.path.basename(f)) for f in filtered_files]
            self.configure(text="\n".join(display_items))
        elif dropped_texts:
            self.file_path = None  # Not a file
            self.configure(text="\n".join(dropped_texts))
        else:
            self.file_path = None
            self.configure(text="Unsupported file type or empty drop")

        if self.on_drop:
            self.on_drop(filtered_files + dropped_texts)

    def _on_drag_leave(self, event):
        if not self.file_path:
            self.configure(text=self.default_text)

    def _parse_drop(self, data):
        items = self.tk.splitlist(data)
        return [f.strip("{}") for f in items]

    def _shorten_name(self, name, max_len=30):
        if len(name) <= max_len:
            return name
        return name[:15] + "..." + name[-10:]

    def _open_file_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All supported files", "*.txt *.pdf *.exe *.png *.jpg *.jpeg *.gif *.bmp *.mp3 *.wav *.ogg *.flac"),
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("Executables", "*.exe"),
                ("Audio files", "*.mp3 *.wav *.ogg *.flac"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.file_path = filepath  # save chosen path
            self.configure(text=os.path.basename(filepath))
            if self.on_drop:
                self.on_drop([filepath])

    def _clear(self, event=None):
        """Clear the label text and reset file path"""
        self.file_path = None
        self.configure(text=self.default_text)
        if self.on_drop:
            self.on_drop([])

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    # --- New public method ---
    def get_file_path(self):
        """Return the last selected/dropped file path, or None if empty."""
        return self.file_path
