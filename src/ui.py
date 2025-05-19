import tkinter as tk
from tkinter import messagebox, filedialog
from recorder import record_audio_thread, play_audio
from file_manager import load_audio_file

def start_ui():
    root = tk.Tk()
    root.title("ToneTrace - Identificador de Efectos de Sonido")
    root.geometry("400x350")
    root.resizable(False, False)
    root.configure(bg="#F0F0F0")

    status_label = tk.Label(root, text="Presiona el micrófono o carga un archivo", font=("Arial", 10), bg="#F0F0F0")
    status_label.pack(pady=20)

    def update_status(msg):
        status_label.config(text=msg)
        root.update()

    tk.Button(root, text="🎤 Grabar Sonido", font=("Arial", 16), bg="#4CAF50", fg="white",
               command=lambda: record_audio_thread(update_status)).pack(pady=15)

    tk.Button(root, text="📂 Cargar Archivo de Audio", font=("Arial", 14), bg="#4CAF50", fg="white",
               command=lambda: load_audio_file(update_status)).pack(pady=10)

    tk.Button(root, text="▶️ Reproducir Último Sonido", font=("Arial", 16), bg="#4CAF50", fg="white",
               command=play_audio).pack(pady=10)

    root.mainloop()
