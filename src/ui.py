"""
TONETRACE - INTERFAZ GRÁFICA PARA IDENTIFICACIÓN DE AUDIO

Este módulo implementa la interfaz gráfica de usuario (GUI) para ToneTrace,
una aplicación de identificación de efectos de sonido basada en análisis espectral
y algoritmos de fingerprinting similares a Shazam.

FLUJO DE TRABAJO DE LA APLICACIÓN:
==================================

1. Usuario graba audio o carga archivo
2. Audio se procesa mediante FFT y detección de picos
3. Se generan huellas digitales (fingerprints)
4. Se comparan con base de datos de audio
5. Se muestran resultados de similitud
6. Visualización opcional de espectrogramas
"""

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
    """
    Función principal que inicializa y ejecuta la interfaz gráfica de ToneTrace.
    
    COMPONENTES PRINCIPALES:
    =======================
    
    - Frame de controles: Botones de grabación, carga y análisis
    - Área de configuración: Duración de grabación
    - Panel de visualización: Espectrogramas y gráficos
    - Lista de resultados: Archivos similares encontrados
    - Barra de estado: Feedback al usuario
    """
    
    # ======== CONFIGURACIÓN INICIAL DE LA INTERFAZ ========
    
    # Configuración del tema visual
    ctk.set_appearance_mode("Dark")        # Modo oscuro para mejor experiencia visual
    ctk.set_default_color_theme("blue")    # Tema azul consistente con branding

    # Creación de la ventana principal
    root = ctk.CTk()
    root.title("ToneTrace - Identificador de Efectos de Sonido")
    root.geometry("950x650")  # Tamaño optimizado para contenido y usabilidad

    # Inicialización del analizador de audio (patrón Singleton implícito)
    analyzer = AudioAnalyzer()

    # ======== CONSTRUCCIÓN DEL LAYOUT PRINCIPAL ========
    
    # Frame contenedor principal con bordes redondeados
    main_frame = ctk.CTkFrame(root, corner_radius=10)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # ======== ÁREA DE CONTROLES Y BOTONES ========
    
    # Frame horizontal para organización de controles
    button_frame = ctk.CTkFrame(main_frame)
    button_frame.pack(fill="x", pady=10)

    # PATRÓN DE AGRUPACIÓN: Frames separados para funcionalidades relacionadas
    
    # Frame exclusivo para controles de grabación (Grabar/Cancelar)
    record_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
    record_frame.pack(side="left", padx=5)

    # Frame exclusivo para controles de reproducción (Play/Stop)
    play_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
    play_frame.pack(side="left", padx=5)

    # Etiqueta de estado para feedback inmediato al usuario
    status_label = ctk.CTkLabel(main_frame, text="Presiona el micrófono o carga un archivo", font=("Arial", 14))
    status_label.pack(pady=10)

    # ======== CONFIGURACIÓN DE DURACIÓN DE GRABACIÓN ========
    
    # Variable de control vinculada al slider
    duration_var = ctk.IntVar(value=10)  # Valor por defecto: 10 segundos

    # Componentes del control de duración con layout horizontal
    ctk.CTkLabel(button_frame, text="⏱ Duración (seg)", font=("Arial", 12)).pack(side="left", padx=(5, 0))
    
    # Slider para selección intuitiva de duración
    duration_slider = ctk.CTkSlider(
        button_frame, 
        from_=2, to=60,           # Rango: 2-60 segundos
        number_of_steps=58,       # Pasos discretos para precisión
        variable=duration_var,    # Vinculación bidireccional
        width=150                 # Ancho optimizado
    )
    duration_slider.pack(side="left", padx=5)
    
    # Label dinámico que muestra el valor actual
    duration_label = ctk.CTkLabel(button_frame, textvariable=duration_var, width=30)
    duration_label.pack(side="left", padx=(0, 10))

    # ======== ÁREA DE VISUALIZACIÓN DE RESULTADOS ========
    
    # Frame dedicado para gráficos y espectrogramas
    spectrum_frame = ctk.CTkFrame(main_frame, height=250)
    spectrum_frame.pack(fill="both", expand=True, pady=10)

    # Título de la sección de resultados
    similarity_label = ctk.CTkLabel(main_frame, text="Audios Similares", font=("Arial", 14, "bold"))
    similarity_label.pack(pady=(10, 5))

    # Container para la lista de resultados
    listbox_frame = ctk.CTkFrame(main_frame)
    listbox_frame.pack(fill="both", expand=True)

    # TextBox para mostrar resultados de similitud con scroll automático
    similarity_listbox = ctk.CTkTextbox(listbox_frame, height=100)
    similarity_listbox.pack(fill="both", expand=True, pady=5, padx=5)

    # ======== FUNCIONES DE UTILIDAD Y CALLBACKS ========

    def update_status(msg):
        """
        Actualiza la barra de estado de forma thread-safe.
        
        Args:
            msg (str): Mensaje de estado a mostrar al usuario
        """
        status_label.configure(text=msg)
        root.update()  # Fuerza actualización inmediata de la UI

    def analyze_current_audio():
        """
        Función principal de análisis de audio con threading no bloqueante.
        
        ALGORITMO DE ANÁLISIS:
        =====================
        
        1. VALIDACIÓN:
           - Verificar existencia del archivo de audio
           - Mostrar advertencia si no hay audio disponible
        
        2. PREPARACIÓN DE UI:
           - Limpiar visualizaciones anteriores
           - Mostrar indicador de progreso
        
        3. ANÁLISIS EN BACKGROUND:
           - Extracción de fingerprints del audio
           - Comparación con base de datos
           - Cálculo de scores de similitud
        
        4. ACTUALIZACIÓN DE RESULTADOS:
           - Thread-safe update de la interfaz
           - Visualización de espectrogramas
           - Lista de archivos similares
        
        MANEJO DE THREADING:
        ===================
        El análisis se ejecuta en un hilo separado para evitar bloquear
        la interfaz de usuario. Se utiliza root.after() para garantizar
        que las actualizaciones de UI ocurran en el hilo principal.
        """
        
        # Validación de entrada
        last_audio_file = get_last_audio_file()
        if not last_audio_file or not os.path.exists(last_audio_file):
            messagebox.showwarning("Sin audio", "No hay ningún audio grabado o cargado para analizar.")
            return

        # Indicador de progreso para el usuario
        update_status("🔍 Analizando audio...")

        # Limpieza de visualizaciones anteriores
        for widget in spectrum_frame.winfo_children():
            widget.destroy()
        similarity_listbox.delete("1.0", "end")

        def analysis_logic():
            """
            Lógica de análisis ejecutada en hilo separado.
            
            PROCESO DE ANÁLISIS:
            ===================
            1. Llamada al analizador de audio
            2. Extracción de fingerprints
            3. Comparación con base de datos
            4. Ordenamiento por similitud
            5. Preparación de resultados para UI
            """
            try:
                # Llamada al algoritmo de análisis (similar a Shazam)
                similar_files = analyzer.find_match_in_all_audios(
                    last_audio_file,
                    block_duration=10,  # Parámetro legacy, no usado en nueva implementación
                    min_matches=2       # Umbral mínimo de coincidencias
                )

                # Extracción del mejor match para visualización comparativa
                if similar_files:
                    match_path, _ = similar_files[0]
                else:
                    match_path = None

                def update_ui():
                    """
                    Actualización thread-safe de la interfaz con resultados.
                    """
                    update_analysis_results(similar_files)
                    
                    # Visualización comparativa de espectrogramas
                    if match_path:
                        analyzer.plot_fft_comparison(last_audio_file, match_path)

                # Programar actualización en el hilo principal de UI
                root.after(0, update_ui)

            except Exception as e:
                # Manejo de errores con feedback al usuario
                root.after(0, lambda e=e: (
                    messagebox.showerror("Error", str(e)),
                    update_status("❌ Error durante el análisis")
                ))
        
        # Ejecutar análisis en hilo separado para no bloquear UI
        threading.Thread(target=analysis_logic).start()

    def update_analysis_results(similar_files):
        """
        Actualiza la lista de resultados con archivos similares encontrados.
        
        PRESENTACIÓN DE RESULTADOS:
        ===========================
        
        1. FILTRADO: Solo muestra archivos con similitud > 0
        2. FORMATO: Nombre del archivo + score de similitud
        3. FEEDBACK: Estado visual según resultados encontrados
        
        Args:
            similar_files: Lista de tuplas (archivo, score_similitud)
        """
        # Limpieza de resultados anteriores
        similarity_listbox.delete("1.0", "end")
        
        # Filtrado de resultados válidos (similitud > 0)
        filtered_files = [(f, s) for f, s in similar_files if s > 0]
        
        if filtered_files:
            # Mostrar archivos similares encontrados
            for file_path, similarity in filtered_files:
                name = os.path.basename(file_path)
                similarity_listbox.insert("end", f"{name} - Similitud: {similarity}\n")
            update_status(f"✅ ¡Coincidencias encontradas!")
        else:
            # Mensaje cuando no hay coincidencias
            update_status("✅ Análisis completado. No se encontraron audios similares.")

    def analyze_audio_thread():
        """
        Wrapper para ejecutar análisis en hilo separado.
        """
        threading.Thread(target=analyze_current_audio).start()

    def start_recording():
        """
        Inicia el proceso de grabación de audio con interfaz dinámica.
        
        CARACTERÍSTICAS:
        ===============
        
        1. BOTÓN DINÁMICO DE CANCELACIÓN:
           - Aparece solo durante la grabación
           - Se autodestruye al finalizar
           - Feedback visual inmediato
        
        2. CALLBACK DE ESTADO:
           - Actualización en tiempo real del progreso
           - Mensajes informativos al usuario
           - Limpieza automática de controles temporales
        
        3. GESTIÓN DE THREADING:
           - Grabación no bloqueante
           - Sincronización thread-safe con UI
        """
        
        def on_cancel_pressed():
            """
            Callback para cancelación de grabación.
            """
            cancel_recording()  # Señal de cancelación al módulo recorder
            
            # Destrucción thread-safe del botón de cancelación
            if cancel_btn.winfo_exists():
                root.after(10, cancel_btn.destroy)

        def on_recording_update(msg):
            """
            Callback para actualizaciones de estado durante grabación.
            
            Args:
                msg (str): Mensaje de estado del proceso de grabación
            """
            update_status(msg)
            
            # Limpieza automática cuando termina la grabación
            if "completada" in msg or "cancelada" in msg or "Error" in msg:
                if cancel_btn.winfo_exists():
                    root.after(10, cancel_btn.destroy)

        # Creación dinámica del botón de cancelación
        cancel_btn = ctk.CTkButton(
            record_frame,
            text="🟥",                    # Icono de parar
            width=30, height=28,          # Tamaño compacto
            fg_color="darkred",           # Color de peligro/alerta
            hover_color="#aa0000",        # Efecto hover más oscuro
            text_color="white",
            command=on_cancel_pressed
        )
        cancel_btn.pack(side="left", padx=(3, 0))
        
        # Actualización inmediata de la interfaz antes de iniciar grabación
        root.update_idletasks()
        
        # Inicio de grabación con delay mínimo para estabilidad de UI
        root.after(150, lambda: record_audio_thread(on_recording_update, duration_var.get()))

    def start_playback():
        
        def on_stop_pressed():
            """
            Callback para detener reproducción.
            """
            stop_playback()  # Señal de parada al módulo recorder
            
            # Destrucción thread-safe del botón de parada
            if stop_btn.winfo_exists():
                root.after(10, stop_btn.destroy)

        def threaded_play():
            try:
                play_audio()  # Función de reproducción del módulo recorder
            finally:
                # Limpieza garantizada del botón de parada
                if stop_btn.winfo_exists():
                    root.after(10, stop_btn.destroy)

        # Creación del botón de parada con diseño consistente
        stop_btn = ctk.CTkButton(
            play_frame,
            text="⏹️",                    # Icono de parar
            width=30, height=28,          # Tamaño consistente con botón de cancelar
            fg_color="darkred",           # Color de alerta consistente
            hover_color="#aa0000",        # Efecto hover
            text_color="white",
            command=on_stop_pressed
        )
        stop_btn.pack(side="left", padx=(3, 0))
        
        # Actualización de UI antes de iniciar reproducción
        root.update_idletasks()

        # Inicio de reproducción en hilo separado
        threading.Thread(target=threaded_play).start()

    # ======== DEFINICIÓN DE BOTONES PRINCIPALES ========    
    # Botón de grabación - Inicia captura de audio
    ctk.CTkButton(
        record_frame, 
        text="🎤 Grabar", 
        command=start_recording
    ).pack(side="left")

    # Botón de carga de archivo - Abre diálogo de selección
    ctk.CTkButton(
        button_frame, 
        text="📂 Cargar Audio", 
        command=lambda: load_audio_file(update_status)  # Callback para actualización de estado
    ).pack(side="left", padx=5)

    # Botón de reproducción - Reproduce último audio
    ctk.CTkButton(
        play_frame, 
        text="▶️ Reproducir", 
        command=start_playback
    ).pack(side="left")

    # Botón de análisis - Ejecuta algoritmo de identificación
    ctk.CTkButton(
        button_frame, 
        text="🔍 Analizar", 
        fg_color="#2271b3",           # Color distintivo para acción principal
        command=analyze_audio_thread  # Wrapper para threading
    ).pack(side="left", padx=5)

    # ======== INICIO DEL LOOP PRINCIPAL DE LA APLICACIÓN ========
    
    # Inicia el loop de eventos de tkinter
    # Bloquea hasta que el usuario cierre la aplicación
    root.mainloop()
