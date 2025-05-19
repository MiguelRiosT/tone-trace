import tkinter as tk
from tkinter import messagebox, filedialog
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import os
from datetime import datetime

DURATION = 5  # Duración de la grabación en segundos
FS = 44100    # Frecuencia de muestreo
OUTPUT_DIR = "assets/audio"

# Asegurar que la carpeta exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"input_sound_{timestamp}.wav")

def record_audio():
    try:
        status_label.config(text="🎙️ Grabando... ¡Habla!")
        root.update()
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()

        output_file = generate_filename()
        wav.write(output_file, FS, recording)

        status_label.config(text=f"✅ Grabación completada.\nArchivo: {output_file}")
        root.last_audio_file = output_file
        root.update()

    except Exception as e:
        messagebox.showerror("Error de Grabación", str(e))

def start_recording_thread():
    threading.Thread(target=record_audio).start()

def play_audio():
    if hasattr(root, 'last_audio_file') and os.path.exists(root.last_audio_file):
        fs, data = wav.read(root.last_audio_file)
        sd.play(data, fs)
        sd.wait()
    else:
        messagebox.showwarning("Archivo no encontrado", "No se encontró ningún audio grabado.")

def load_audio_file():
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de audio",
        filetypes=[("Archivos de audio", "*.wav *.mp3")]
    )
    if file_path:
        status_label.config(text=f"📂 Archivo cargado:\n{os.path.basename(file_path)}")
        root.last_audio_file = file_path  # Permite reproducir el archivo cargado

# Configuración de la ventana principal
root = tk.Tk()
root.title("ToneTrace - Identificador de Efectos de Sonido")
root.geometry("400x350")
root.resizable(False, False)

# Botón de grabar
mic_button = tk.Button(root, text="🎤 Grabar Sonido", font=("Arial", 16), command=start_recording_thread)
mic_button.pack(pady=15)

# Botón de cargar archivo
load_button = tk.Button(root, text="📂 Cargar Archivo de Audio", font=("Arial", 14), command=load_audio_file)
load_button.pack(pady=10)

# Botón de reproducir
play_button = tk.Button(root, text="▶️ Reproducir Último Sonido", font=("Arial", 16), command=play_audio)
play_button.pack(pady=10)

# Etiqueta de estado
status_label = tk.Label(root, text="Presiona el micrófono o carga un archivo", font=("Arial", 10))
status_label.pack(pady=20)

root.mainloop()