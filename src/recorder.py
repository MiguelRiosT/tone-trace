from tkinter import messagebox
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import os
from datetime import datetime

DURATION = 5
FS = 44100
OUTPUT_DIR = "assets/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
last_audio_file = None

def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"input_sound_{timestamp}.wav")

def record_audio(update_status):
    global last_audio_file
    try:
        update_status("🎙️ Grabando... ¡Habla!")
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()
        output_file = generate_filename()
        wav.write(output_file, FS, recording)
        last_audio_file = output_file
        update_status(f"✅ Grabación completada.\nArchivo: {output_file}")
    except Exception as e:
        update_status(f"❌ Error: {e}")

def record_audio_thread(update_status):
    threading.Thread(target=record_audio, args=(update_status,)).start()

def play_audio():
    global last_audio_file
    if last_audio_file and os.path.exists(last_audio_file):
        fs, data = wav.read(last_audio_file)
        sd.play(data, fs)
        sd.wait()
    else:
        messagebox.showwarning("Archivo no encontrado", "No se encontró ningún audio grabado.")