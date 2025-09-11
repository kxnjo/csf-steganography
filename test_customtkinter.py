import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES


# callback function
def button_callback():
    print("button is pressed!", flush=True)

def warmup():
    print("warm up done!", flush=True)

# for select file
def select_file():
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Image files", "*.png *.jpg *.bmp"),
                   ("Audio files", "*.wav"),
                   ("All files", "*.*")]
    )
    if file_path:
        messagebox.showinfo("File Selected", f"You selected:\n{file_path}")
        label.configure(text=f"Selected: {file_path}")

# for Drag and Drop (DND)
def drop(event):
    file_path = event.data.strip("{}")   # remove braces on macOS paths
    messagebox.showinfo("File Dropped", f"You dropped:\n{file_path}")
    label.configure(text=f"Dropped: {file_path}")

# ---- GUI Setup ----
ctk.set_appearance_mode("System")   # "Dark", "Light", or "System"
ctk.set_default_color_theme("blue") # "blue", "green", "dark-blue"

# initialise customtkinter
# Important: Use TkinterDnD.Tk instead of ctk.CTk
app = TkinterDnD.Tk()
app.title("Steganography")
app.geometry("500x300") # define the window size

# include the button
button = ctk.CTkButton(app, text = "click me!", command = button_callback)
button.pack(pady=20)  # <-- this makes it visible

# testing file upload
label = ctk.CTkLabel(app, text="No file selected yet", wraplength=400)
label.pack(pady=20)

upload_btn = ctk.CTkButton(app, text="Upload File", command=select_file)
upload_btn.pack(pady=10)

# drag and drop
label = ctk.CTkLabel(app, text="Drag a file here", width=400, height=100)
label.pack(pady=40)

# Register label as a drop target
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', drop)

# schedule warmup to run after 100ms (0.1 sec)
app.after(100, warmup)

# launch the application?
app.mainloop()