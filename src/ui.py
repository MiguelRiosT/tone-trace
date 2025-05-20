import tkinter as tk
from tkinter import messagebox
from recorder import record_audio_thread, play_audio, get_last_audio_file
from file_manager import load_audio_file
from audio_analyzer import AudioAnalyzer
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np

def start_ui():
    root = tk.Tk()
    root.title("ToneTrace - Identificador de Efectos de Sonido")
    root.geometry("800x600")
    root.resizable(True, True)
    root.configure(bg="#F0F0F0")

    # Crear el analizador de audio
    analyzer = AudioAnalyzer()

    # Frame principal
    main_frame = tk.Frame(root, bg="#F0F0F0")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Frame para los botones
    button_frame = tk.Frame(main_frame, bg="#F0F0F0")
    button_frame.pack(fill=tk.X, pady=10)

    status_label = tk.Label(main_frame, text="Presiona el micrófono o carga un archivo", 
                           font=("Arial", 10), bg="#F0F0F0", wraplength=700)
    status_label.pack(pady=10)

    # Frame para resultados
    results_frame = tk.Frame(main_frame, bg="#F0F0F0")
    results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    # Área para mostrar el espectro
    spectrum_frame = tk.Frame(results_frame, bg="#F0F0F0")
    spectrum_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    # Área para mostrar resultados de similitud
    similarity_frame = tk.Frame(results_frame, bg="#F0F0F0")
    similarity_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    # Lista para mostrar resultados de similitud
    similarity_list = tk.Listbox(similarity_frame, font=("Arial", 10), bg="white", 
                                selectmode=tk.SINGLE, height=5)
    similarity_list.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Scrollbar para la lista
    scrollbar = tk.Scrollbar(similarity_list)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    similarity_list.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=similarity_list.yview)

    def update_status(msg):
        status_label.config(text=msg)
        root.update()

    def analyze_current_audio():
        last_audio_file = get_last_audio_file()
        if not last_audio_file or not os.path.exists(last_audio_file):
            messagebox.showwarning("Sin audio", "No hay ningún audio grabado o cargado para analizar.")
            return

        update_status("🔍 Analizando audio...")

        # Limpiar resultados anteriores
        for widget in spectrum_frame.winfo_children():
            widget.destroy()
        similarity_list.delete(0, tk.END)

        try:
            # Cargar y analizar el audio
            signal, sr = analyzer.load_audio(last_audio_file)
            if signal is None:
                messagebox.showerror("Error", "No se pudo cargar el archivo de audio.")
                return

            # Calcular FFT
            freqs, fft_result = analyzer.compute_fft(signal)
            if freqs is None or fft_result is None:
                messagebox.showerror("Error", "No se pudo analizar el espectro de frecuencia.")
                return

            # Buscar audios similares
            similar_files = analyzer.find_similar_audio(last_audio_file)

            # Actualizar la interfaz con los resultados
            root.after(0, lambda: update_analysis_results(signal, sr, freqs, fft_result, similar_files))

        except Exception as e:
            messagebox.showerror("Error", f"Error durante el análisis: {str(e)}")
            update_status("❌ Error durante el análisis")

    def update_analysis_results(signal, sr, freqs, fft_result, similar_files):
        # --- ACTUALIZACIÓN DE LA INTERFAZ EN HILO PRINCIPAL ---
        # Mostrar el espectro
        tiempos = np.linspace(0, len(signal) / sr, num=len(signal))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tiempos, signal, color='dodgerblue')
        ax.set_title("Forma de onda del audio")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=spectrum_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        # Mostrar resultados de similitud
        if similar_files:
            update_status(f"✅ Análisis completado. Se encontraron {len(similar_files)} audios similares.")
            for file_path, similarity in similar_files:
                file_name = os.path.basename(file_path)
                similarity_list.insert(tk.END, f"{file_name} - Similitud: {similarity:.2f}")

            def play_selected():
                selection = similarity_list.curselection()
                if selection:
                    index = selection[0]
                    file_path = similar_files[index][0]
                    try:
                        signal, sr = analyzer.load_audio(file_path)
                        if signal is not None:
                            import sounddevice as sd
                            sd.play(signal, sr)
                            sd.wait()
                    except Exception as e:
                        messagebox.showerror("Error", f"Error al reproducir el audio: {e}")

            play_button = tk.Button(similarity_frame, text="▶️ Reproducir Audio Seleccionado",
                                   font=("Arial", 10), bg="#4CAF50", fg="white",
                                   command=play_selected)
            play_button.pack(pady=5)
        else:
            update_status("✅ Análisis completado. No se encontraron audios similares.")

    def analyze_audio_thread():
        threading.Thread(target=analyze_current_audio).start()

    # Botones
    tk.Button(button_frame, text="🎤 Grabar Sonido", font=("Arial", 12), bg="#4CAF50", fg="white",
               command=lambda: record_audio_thread(update_status)).pack(side=tk.LEFT, padx=5)

    tk.Button(button_frame, text="📂 Cargar Archivo de Audio", font=("Arial", 12), bg="#4CAF50", fg="white",
               command=lambda: load_audio_file(update_status)).pack(side=tk.LEFT, padx=5)

    tk.Button(button_frame, text="▶️ Reproducir Último Sonido", font=("Arial", 12), bg="#4CAF50", fg="white",
               command=play_audio).pack(side=tk.LEFT, padx=5)
               
    tk.Button(button_frame, text="🔍 Analizar Audio", font=("Arial", 12), bg="#2196F3", fg="white",
               command=analyze_audio_thread).pack(side=tk.LEFT, padx=5)

    # Etiqueta para el espectro
    tk.Label(spectrum_frame, text="Espectro de Frecuencia", font=("Arial", 12, "bold"), 
            bg="#F0F0F0").pack(pady=5)
            
    # Etiqueta para resultados de similitud
    tk.Label(similarity_frame, text="Audios Similares", font=("Arial", 12, "bold"), 
            bg="#F0F0F0").pack(pady=5)

    root.mainloop()

    last_audio_file = None
