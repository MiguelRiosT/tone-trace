from tkinter import filedialog
import os
import recorder

def load_audio_file(update_status):
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de audio",
        filetypes=[("Archivos de audio", "*.wav *.mp3")]
    )
    if file_path:
        update_status(f"📂 Archivo cargado:\n{os.path.basename(file_path)}")
        recorder.last_audio_file = file_path