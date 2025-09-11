import customtkinter as ctk

# callback function
def button_callback():
    print("button is pressed!", flush=True)

def warmup():
    print("warm up done!", flush=True)

# initialise customtkinter
app = ctk.CTk()
app.title("Steganography")
app.geometry("400x150") # define the window size

# include the button
button = ctk.CTkButton(app, text = "click me!", command = button_callback)
button.pack(pady=20)  # <-- this makes it visible

# schedule warmup to run after 100ms (0.1 sec)
app.after(100, warmup)

# launch the application?
app.mainloop()