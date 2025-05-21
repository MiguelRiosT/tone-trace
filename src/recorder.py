"""
Módulo especializado en la captura, procesamiento y reproducción de audio.
Este componente es fundamental para el sistema de análisis de Fourier ya que
proporciona las señales de audio que serán posteriormente analizadas espectralmente.

FUNCIONALIDADES PRINCIPALES:
1. Grabación de audio desde micrófono con calidad profesional
2. Reproducción de archivos de audio
3. Gestión de archivos temporales y permanentes
4. Control de operaciones en tiempo real (cancelar/detener)
5. Integración con el sistema de análisis espectral

PARÁMETROS DE AUDIO OPTIMIZADOS PARA ANÁLISIS DE FOURIER:
- Frecuencia de muestreo: 44100 Hz (estándar de audio digital)
- Resolución: 16 bits (int16)
- Canales: Mono (1 canal) - simplifica el análisis espectral
"""

from tkinter import messagebox, Toplevel, Entry, Label, Button
import sounddevice as sd  # Biblioteca profesional para audio en tiempo real
import numpy as np        # Operaciones numéricas para procesamiento de señales
import scipy.io.wavfile as wav  # Lectura/escritura de archivos WAV
import threading         # Operaciones asíncronas para no bloquear UI
import os
from datetime import datetime
from audio_analyzer import AudioAnalyzer

# ======== CONFIGURACIÓN GLOBAL DE AUDIO ========
"""
Parámetros optimizados para análisis de Fourier:

DURATION (10 seg): Tiempo suficiente para capturar patrones espectrales significativos
FS (44100 Hz): Frecuencia de Nyquist permite capturar frecuencias hasta 22050 Hz
               (rango completo de audición humana: 20Hz - 20kHz)
"""
DURATION = 10    # Duración por defecto de grabación (segundos)
FS = 44100       # Frecuencia de muestreo (Hz) - Estándar profesional
OUTPUT_DIR = "assets/audio"  # Directorio para archivos permanentes
TEMP_DIR = "assets/temp"     # Directorio para archivos temporales

# Crear estructura de directorios si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ======== VARIABLES GLOBALES DE ESTADO ========
last_audio_file = None        # Referencia al último archivo grabado/cargado
is_recording_cancelled = False # Flag para control de cancelación de grabación

def generate_filename(custom_name=None):
    """
    Genera nombres de archivo únicos y seguros para el sistema operativo.
    """
    if custom_name:
        # Sanitización: solo permite caracteres alfanuméricos y algunos especiales
        custom_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '_', '-'))
        return os.path.join(OUTPUT_DIR, f"{custom_name}.wav")
    
    # Timestamp con formato YYYYMMDD_HHMMSS para ordenación cronológica
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"input_sound_{timestamp}.wav")

def record_audio(update_status, duration):
    """
    Función principal de grabación de audio con monitoreo en tiempo real.
    
    Args:
        update_status (callable): Función callback para actualizar UI
        duration (int): Duración de grabación en segundos
        
    PROCESO DE GRABACIÓN:
    1. Inicializa el buffer de grabación con parámetros optimizados
    2. Captura audio en tiempo real desde el micrófono
    3. Monitorea cancelaciones cada 0.1 segundos
    4. Guarda el archivo en formato WAV sin compresión
    5. Actualiza referencias globales para análisis posterior
    """
    global last_audio_file, is_recording_cancelled
    is_recording_cancelled = False
    
    try:
        update_status(f"🎙️ Grabando por {duration} segundos... ¡Habla!")
        
        # Configuración del buffer de grabación
        # int16: rango -32768 a 32767, suficiente resolución para análisis FFT
        recording = sd.rec(
            int(duration * FS),  # Número total de muestras
            samplerate=FS,       # Frecuencia de muestreo
            channels=1,          # Mono para simplificar análisis
            dtype='int16'        # 16 bits de resolución
        )
        
        # Monitoreo de cancelación cada 100ms
        # Permite cancelación responsiva sin afectar calidad de audio
        for i in range(duration * 10):  # 10 checks por segundo
            if is_recording_cancelled:
                sd.stop()  # Detiene inmediatamente la grabación
                update_status("❌ Grabación cancelada.")
                return
            sd.sleep(100)  # 100ms de espera
        
        # Generación de archivo de salida
        output_file = generate_filename()
        
        # Guardado en formato WAV sin compresión
        # WAV preserva toda la información espectral necesaria para FFT
        wav.write(output_file, FS, recording)
        
        # Actualiza referencia global para otros módulos
        last_audio_file = output_file
        update_status(f"✅ Grabación completada.\nArchivo: {output_file}")
        
    except Exception as e:
        # Manejo robusto de errores con feedback específico
        update_status(f"❌ Error: {e}")
    
    finally:
        # Limpieza de archivos temporales (si existen)
        # Nota: variable 'temp_file' no está definida en el código actual
        # Esto parece ser código residual de una versión anterior
        if 'temp_file' in locals() and temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass  # Ignora errores de limpieza

def cancel_recording():
    """
    Función de cancelación segura para grabaciones en curso.
    
    IMPORTANCIA:
    - Evita archivos corruptos o incompletos
    - Mantiene integridad del sistema de archivos
    - Proporciona control responsivo al usuario
    """
    global is_recording_cancelled
    is_recording_cancelled = True

def record_audio_thread(update_status, duration):
    """
    Wrapper para ejecutar grabación en hilo separado.
    """
    threading.Thread(target=record_audio, args=(update_status, duration)).start()

def play_audio():
    """
    Reproduce el último archivo de audio grabado o cargado.
    """
    global last_audio_file
    
    if last_audio_file and os.path.exists(last_audio_file):
        # Lee archivo WAV completo en memoria
        fs, data = wav.read(last_audio_file)
        
        # Reproduce con parámetros originales
        sd.play(data, fs)
        sd.wait()  # Bloquea hasta completar reproducción
    else:
        messagebox.showwarning("Archivo no encontrado", "No se encontró ningún audio grabado.")

def get_last_audio_file():
    """
    Accessor para obtener referencia al último archivo de audio.
    """
    return last_audio_file

def stop_playback():
    """
    Detiene inmediatamente cualquier reproducción en curso.
    """
    try:
        sd.stop()  # Detiene inmediatamente cualquier operación de audio
    except Exception as e:
        # Log de error sin interrumpir funcionamiento
        print(f"Error al detener reproducción: {e}")
