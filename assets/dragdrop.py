import customtkinter as ctk
from tkinter import filedialog, Menu
import os

class DragDropLabel(ctk.CTkLabel):
    ACCEPTED_EXTENSIONS = (
        ".txt", ".pdf", ".exe", ".png", ".jpg", ".jpeg", ".gif", ".bmp",
        ".mp3", ".wav", ".ogg", ".flac"
    )

    def __init__(self, parent, text="Drag & Drop or Click to Select File", **kwargs):
        super().__init__(parent, text=text, **kwargs)
        self.default_text = text
        self.file_path = None
        self.on_file_selected = None  # user sets this externally

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

    # --- Drag-drop handler ---
    def _on_drop(self, event):
        files = [f for f in self.tk.splitlist(event.data) if os.path.isfile(f)]
        filtered_files = [f for f in files if f.lower().endswith(self.ACCEPTED_EXTENSIONS)]
        if filtered_files:
            self.file_path = filtered_files[0]
            self.configure(text=os.path.basename(self.file_path))
        else:
            self.file_path = None
            self.configure(text="Unsupported file type or empty drop")

        if callable(self.on_file_selected):
            self.on_file_selected(self.file_path)

    def _on_drag_leave(self, event):
        if not self.file_path:
            self.configure(text=self.default_text)

    # --- Browse dialog ---
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
            self.file_path = filepath
            self.configure(text=os.path.basename(filepath))
            if callable(self.on_file_selected):
                self.on_file_selected(self.file_path)  # <- now triggers preview

    def _clear(self, event=None):
        self.file_path = None
        self.configure(text=self.default_text)
        if callable(self.on_file_selected):
            self.on_file_selected(self.file_path)

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    # --- Public method ---
    def get_file_path(self):
        return self.file_path
