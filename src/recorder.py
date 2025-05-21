from tkinter import messagebox
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import os
from datetime import datetime

DURATION = 10
FS = 44100
OUTPUT_DIR = "assets/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
last_audio_file = None
is_recording_cancelled = False

def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"input_sound_{timestamp}.wav")

def record_audio(update_status, duration):
    global last_audio_file, is_recording_cancelled
    is_recording_cancelled = False
    try:
        update_status(f"🎙️ Grabando por {duration} segundos... ¡Habla!")
        recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='int16')
        for i in range(duration * 10):  # chequeo cada 0.1 seg
            if is_recording_cancelled:
                sd.stop()
                update_status("❌ Grabación cancelada.")
                return
            sd.sleep(100)

        output_file = generate_filename()
        wav.write(output_file, FS, recording)
        last_audio_file = output_file
        update_status(f"✅ Grabación completada.\nArchivo: {output_file}")
    except Exception as e:
        update_status(f"❌ Error: {e}")

def cancel_recording():
    global is_recording_cancelled
    is_recording_cancelled = True

def record_audio_thread(update_status, duration):
    threading.Thread(target=record_audio, args=(update_status, duration)).start()

def play_audio():
    global last_audio_file
    if last_audio_file and os.path.exists(last_audio_file):
        fs, data = wav.read(last_audio_file)
        sd.play(data, fs)
        sd.wait()
    else:
        messagebox.showwarning("Archivo no encontrado", "No se encontró ningún audio grabado.")

def get_last_audio_file():
    return last_audio_file

def stop_playback():
    try:
        sd.stop()
    except Exception as e:
        print(f"Error al detener reproducción: {e}")