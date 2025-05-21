from tkinter import messagebox, Toplevel, Entry, Label, Button
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import os
from datetime import datetime
from audio_analyzer import AudioAnalyzer

DURATION = 10
FS = 44100
OUTPUT_DIR = "assets/audio"
TEMP_DIR = "assets/temp"
# Crear ambos directorios si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
last_audio_file = None
is_recording_cancelled = False

def generate_filename(custom_name=None):
    if custom_name:
        # Asegurarse de que el nombre sea válido para un archivo
        custom_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '_', '-'))
        return os.path.join(OUTPUT_DIR, f"{custom_name}.wav")
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
    finally:
        # Limpiar archivo temporal
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

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