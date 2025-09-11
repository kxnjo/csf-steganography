import customtkinter as ctk

class DragDropLabel(ctk.CTkLabel):
    """
    A reusable drag-and-drop label widget for CustomTkinter.
    Supports file dropping from OS file explorer.
    Auto-resets text if drag leaves without dropping.
    """

    def __init__(self, parent, text="Drag & Drop File Here", on_drop=None, **kwargs):
        super().__init__(parent, text=text, **kwargs)

        self.default_text = text
        self.on_drop = on_drop  # Callback when file is dropped
        self.configure(corner_radius=10, fg_color="gray20", width=250, height=120)

        # Enable drop events (TkinterDnD2 API)
        self.drop_target_register("DND_Files")
        self.dnd_bind("<<Drop>>", self._on_drop)
        self.dnd_bind("<<DragLeave>>", self._on_drag_leave)

    def _on_drop(self, event):
        # Extract dropped file path(s)
        files = self._parse_drop(event.data)

        # Replace label text with new filenames only
        filenames = [f.split("/")[-1].split("\\")[-1] for f in files]
        self.configure(text="\n".join(filenames))

        # If a user-defined callback exists, pass full file paths
        if self.on_drop:
            self.on_drop(files)

    def _on_drag_leave(self, event):
        """Reset label text if drag leaves without dropping."""
        self.configure(text=self.default_text)

    def _parse_drop(self, data):
        """
        Handles OS-specific drop data formatting.
        On Windows, files come in {path1} {path2} format.
        """
        files = self.tk.splitlist(data)
        return [f.strip("{}") for f in files]
