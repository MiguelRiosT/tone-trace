from tkinter import messagebox, Toplevel, Entry, Label, Button
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import os
from datetime import datetime
from audio_analyzer import AudioAnalyzer

DURATION = 5
FS = 44100
OUTPUT_DIR = "assets/audio"
TEMP_DIR = "assets/temp"
# Crear ambos directorios si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
last_audio_file = None
analyzer = AudioAnalyzer()

def generate_filename(custom_name=None):
    if custom_name:
        # Asegurarse de que el nombre sea válido para un archivo
        custom_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '_', '-'))
        return os.path.join(OUTPUT_DIR, f"{custom_name}.wav")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"input_sound_{timestamp}.wav")

def generate_temp_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(TEMP_DIR, f"temp_recording_{timestamp}.wav")

def cleanup_temp_files():
    """Limpia archivos temporales antiguos"""
    try:
        for file in os.listdir(TEMP_DIR):
            if file.endswith('.wav'):
                os.remove(os.path.join(TEMP_DIR, file))
    except Exception as e:
        print(f"Error limpiando archivos temporales: {e}")

def ask_filename(root_window):
    result = [None]
    
    def on_ok():
        result[0] = entry.get()
        dialog.destroy()
    
    def on_cancel():
        dialog.destroy()
    
    # Crear ventana de diálogo
    dialog = Toplevel(root_window)
    dialog.title("Guardar Audio")
    dialog.geometry("300x150")
    dialog.transient(root_window)
    dialog.grab_set()
    
    # Centrar la ventana
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (root_window.winfo_width() // 2) - (width // 2)
    y = (root_window.winfo_height() // 2) - (height // 2)
    dialog.geometry(f'+{x}+{y}')
    
    # Agregar widgets
    Label(dialog, text="¿Con qué nombre deseas guardar el audio?", 
          wraplength=250, pady=10).pack()
    
    entry = Entry(dialog)
    entry.insert(0, "mi_audio")
    entry.pack(pady=10, padx=20)
    entry.select_range(0, 'end')
    entry.focus()
    
    # Frame para botones
    button_frame = Label(dialog)
    button_frame.pack(pady=10)
    
    Button(button_frame, text="Aceptar", command=on_ok).pack(side='left', padx=10)
    Button(button_frame, text="Cancelar", command=on_cancel).pack(side='left')
    
    # Esperar hasta que se cierre la ventana
    dialog.wait_window()
    return result[0]

def record_audio(update_status, root_window):
    global last_audio_file
    temp_file = None
    try:
        # Limpiar archivos temporales antiguos
        cleanup_temp_files()
        
        update_status("🎙️ Grabando... ¡Habla!")
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()
        
        # Primero guardamos temporalmente el audio para analizarlo
        temp_file = generate_temp_filename()
        wav.write(temp_file, FS, recording)
        
        # Analizamos si existe un audio similar en la carpeta de audios permanentes
        similar_files = []
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith('.wav'):
                file_path = os.path.join(OUTPUT_DIR, file)
                similarity = analyzer.compare_audio_files(temp_file, file_path)
                if similarity > 0.85:  # Umbral de similitud
                    similar_files.append((file_path, similarity))
        
        if similar_files:
            # Si encontramos audios similares
            most_similar_file = os.path.basename(similar_files[0][0])
            similarity_value = similar_files[0][1]
            last_audio_file = similar_files[0][0]  # Usamos el archivo original, no el temporal
            update_status(f"⚠️ Audio similar encontrado: {most_similar_file} (Similitud: {similarity_value:.2f})")
        else:
            # Si no hay audios similares, procedemos a guardar el nuevo audio
            update_status("✨ Nuevo audio detectado. Guardando...")
            
            # Solicitar al usuario el nombre del archivo
            custom_name = ask_filename(root_window)
            
            if custom_name:
                output_file = generate_filename(custom_name)
            else:
                output_file = generate_filename()
            
            # Copiamos el archivo temporal al definitivo
            wav.write(output_file, FS, recording)
            last_audio_file = output_file
            update_status(f"✅ Nuevo audio guardado como: {os.path.basename(output_file)}")
        
    except Exception as e:
        update_status(f"❌ Error: {e}")
    finally:
        # Limpiar archivo temporal
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def record_audio_thread(update_status, root_window):
    threading.Thread(target=record_audio, args=(update_status, root_window)).start()

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
