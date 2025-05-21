import customtkinter as ctk
import os
from tkinter import messagebox
from recorder import record_audio_thread, play_audio, get_last_audio_file, cancel_recording, stop_playback
from file_manager import load_audio_file
from audio_analyzer import AudioAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import librosa
import threading

def start_ui():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("ToneTrace - Identificador de Efectos de Sonido")
    root.geometry("950x650")

    analyzer = AudioAnalyzer()

    # ======== Layout principal ========
    main_frame = ctk.CTkFrame(root, corner_radius=10)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # ======== Área de botones ========
    button_frame = ctk.CTkFrame(main_frame)
    button_frame.pack(fill="x", pady=10)

    # Frame exclusivo para grabar y cancelar
    record_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
    record_frame.pack(side="left", padx=5)

    # Frame exclusivo para reproducir y detener
    play_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
    play_frame.pack(side="left", padx=5)

    status_label = ctk.CTkLabel(main_frame, text="Presiona el micrófono o carga un archivo", font=("Arial", 14))
    status_label.pack(pady=10)

    # ======== Duración de grabación ========
    duration_var = ctk.IntVar(value=10)

    ctk.CTkLabel(button_frame, text="⏱ Duración (seg)", font=("Arial", 12)).pack(side="left", padx=(5, 0))
    duration_slider = ctk.CTkSlider(button_frame, from_=2, to=60, number_of_steps=58, variable=duration_var, width=150)
    duration_slider.pack(side="left", padx=5)
    duration_label = ctk.CTkLabel(button_frame, textvariable=duration_var, width=30)
    duration_label.pack(side="left", padx=(0, 10))

    # ======== Área de resultados ========
    spectrum_frame = ctk.CTkFrame(main_frame, height=250)
    spectrum_frame.pack(fill="both", expand=True, pady=10)

    similarity_label = ctk.CTkLabel(main_frame, text="Audios Similares", font=("Arial", 14, "bold"))
    similarity_label.pack(pady=(10, 5))

    listbox_frame = ctk.CTkFrame(main_frame)
    listbox_frame.pack(fill="both", expand=True)

    similarity_listbox = ctk.CTkTextbox(listbox_frame, height=100)
    similarity_listbox.pack(fill="both", expand=True, pady=5, padx=5)

    def update_status(msg):
        status_label.configure(text=msg)
        root.update()

    def analyze_current_audio():
        last_audio_file = get_last_audio_file()
        if not last_audio_file or not os.path.exists(last_audio_file):
            messagebox.showwarning("Sin audio", "No hay ningún audio grabado o cargado para analizar.")
            return

        update_status("🔍 Analizando audio...")

        for widget in spectrum_frame.winfo_children():
            widget.destroy()
        similarity_listbox.delete("1.0", "end")

        def analysis_logic():
            try:
                similar_files = analyzer.find_match_in_all_audios(
                    last_audio_file,
                    block_duration=10,  # Valor dummy, no se usa en el nuevo backend
                    min_matches=2
                )

                if similar_files:
                    match_path, _ = similar_files[0]
                else:
                    match_path = None

                def update_ui():
                    update_analysis_results(similar_files)
                    if match_path:
                        analyzer.plot_fft_comparison(last_audio_file, match_path)

                root.after(0, update_ui)

            except Exception as e:
                root.after(0, lambda e=e: (
                    messagebox.showerror("Error", str(e)),
                    update_status("❌ Error durante el análisis")
                ))
        threading.Thread(target=analysis_logic).start()

    def update_analysis_results(similar_files):
        similarity_listbox.delete("1.0", "end")
        filtered_files = [(f, s) for f, s in similar_files if s > 0]
        if filtered_files:
            for file_path, similarity in filtered_files:
                name = os.path.basename(file_path)
                similarity_listbox.insert("end", f"{name} - Similitud: {similarity}\n")
            update_status(f"✅ ¡Coincidencias encontradas!")
        else:
            update_status("✅ Análisis completado. No se encontraron audios similares.")

    def analyze_audio_thread():
        threading.Thread(target=analyze_current_audio).start()

    def start_recording():
        def on_cancel_pressed():
            cancel_recording()
            if cancel_btn.winfo_exists():
                root.after(10, cancel_btn.destroy)

        def on_recording_update(msg):
            update_status(msg)
            if "completada" in msg or "cancelada" in msg or "Error" in msg:
                if cancel_btn.winfo_exists():
                    root.after(10, cancel_btn.destroy)

        cancel_btn = ctk.CTkButton(
            record_frame,
            text="🟥",
            width=30,
            height=28,
            fg_color="darkred",
            hover_color="#aa0000",
            text_color="white",
            command=on_cancel_pressed
        )
        cancel_btn.pack(side="left", padx=(3, 0))
        root.update_idletasks()
        root.after(150, lambda: record_audio_thread(on_recording_update, duration_var.get()))

    def start_playback():
        def on_stop_pressed():
            stop_playback()
            if stop_btn.winfo_exists():
                root.after(10, stop_btn.destroy)

        def threaded_play():
            try:
                play_audio()
            finally:
                if stop_btn.winfo_exists():
                    root.after(10, stop_btn.destroy)

        stop_btn = ctk.CTkButton(
            play_frame,
            text="⏹️",
            width=30,
            height=28,
            fg_color="darkred",
            hover_color="#aa0000",
            text_color="white",
            command=on_stop_pressed
        )
        stop_btn.pack(side="left", padx=(3, 0))
        root.update_idletasks()

        threading.Thread(target=threaded_play).start()

    # ======== Botones funcionales ========
    ctk.CTkButton(
        record_frame, text="🎤 Grabar", command=start_recording
    ).pack(side="left")

    ctk.CTkButton(
        button_frame, text="📂 Cargar Audio", command=lambda: load_audio_file(update_status)
    ).pack(side="left", padx=5)

    ctk.CTkButton(
        play_frame, text="▶️ Reproducir", command=start_playback
    ).pack(side="left")

    ctk.CTkButton(
        button_frame, text="🔍 Analizar", fg_color="#2271b3", command=analyze_audio_thread
    ).pack(side="left", padx=5)

    root.mainloop()