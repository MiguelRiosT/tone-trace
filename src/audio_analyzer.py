"""
ANÁLISIS DE AUDIO CON FUNDAMENTOS MATEMÁTICOS - ALGORITMO TIPO SHAZAM

Este módulo implementa un sistema de reconocimiento de audio basado en fingerprinting,
similar al algoritmo utilizado por Shazam. Utiliza la Transformada Rápida de Fourier (FFT)
para analizar el contenido frecuencial del audio y generar "huellas digitales" únicas.

FUNDAMENTOS MATEMÁTICOS:
========================

1. TRANSFORMADA DE FOURIER:
   La Transformada de Fourier descompone una señal temporal f(t) en sus componentes
   frecuenciales. Matemáticamente:
   
   F(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt
   
   En el caso discreto (DFT), trabajamos con muestras:
   X[k] = Σ_{n=0}^{N-1} x[n] e^{-i2πkn/N}
   
   donde:
   - x[n] son las muestras de audio
   - X[k] son los coeficientes frecuenciales
   - N es el tamaño de la ventana

2. ALGORITMO DE SHAZAM:
   Shazam funciona mediante estos pasos:
   a) Conversión a espectrograma (tiempo-frecuencia)
   b) Detección de picos espectrales
   c) Generación de hashes de constelaciones
   d) Comparación con base de datos
   
   La clave está en crear "constelaciones" de picos que son invariantes
   al ruido y distorsión.

PARÁMETROS CLAVE DEL ALGORITMO:
===============================
"""

import os
import numpy as np
import hashlib
from pathlib import Path
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
import matplotlib.pyplot as plt
import librosa
from numpy.fft import fft, fftfreq

# Configuración de directorio de audio
AUDIO_DIR = 'assets/audio'

# Índices para las tuplas de picos (frecuencia, tiempo)
IDX_FREQ_I = 0  # Índice de frecuencia en el pico
IDX_TIME_J = 1  # Índice de tiempo en el pico

# PARÁMETROS DE LA TRANSFORMADA DE FOURIER
FS_DEFAULT = 44100  # Frecuencia de muestreo estándar (Hz)
                        # Según el teorema de Nyquist: fs ≥ 2*f_max
                        # Para audio humano (20Hz-20kHz), 44.1kHz es suficiente

EFFECT_WINDOW_SIZE = 1024  # Tamaño de ventana para FFT
                                   # Mayor ventana = mejor resolución frecuencial
                                   # Menor ventana = mejor resolución temporal
                                   # Compromiso tiempo-frecuencia de Heisenberg: Δt·Δf ≥ 1/4π

RELATIONSHIP_OVERLAP_BY_DEFECT = 0.5  # 50% de superposición entre ventanas
                                           # Evita pérdida de información en los bordes

# PARÁMETROS DEL ALGORITMO DE FINGERPRINTING
DEFAULT_FAN_VALUE = 15        # Número de picos futuros a considerar para hashes
AMP_MIN_DEFECT = 20      # Tamaño del filtro para detección de picos locales

# PARÁMETROS DE GENERACIÓN DE HASHES (CONSTELACIONES)
MIN_DELTA_TIME_HASH = 0         # Mínima diferencia temporal entre picos
MAX_DELTA_TIME_HASH = 200       # Máxima diferencia temporal entre picos
ORDER_PICES = True              # Ordenar picos cronológicamente
FOOTPRINT_REDUCTION = 20             # Caracteres del hash a conservar


class AudioAnalyzer:
    """
    Analizador de audio que implementa un algoritmo de fingerprinting similar a Shazam.
    
    El proceso completo sigue estos pasos matemáticos:
    1. Segmentación del audio en ventanas temporales
    2. Aplicación de ventana de Hanning para reducir efectos de borde
    3. Cálculo de FFT para obtener espectro de frecuencias
    4. Conversión a escala logarítmica (dB)
    5. Detección de picos espectrales locales
    6. Generación de hashes basados en constelaciones de picos
    7. Comparación de hashes para encontrar coincidencias
    """
    
    def __init__(self, audio_dir=AUDIO_DIR):
        """
        Inicializa el analizador de audio.
        
        Args:
            audio_dir: Directorio donde se almacenan los archivos de audio
        """
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._ya_graficado = False  # Control para mostrar espectrograma solo una vez

    def _reemplazarCeros(self, datos):
        """
        Reemplaza valores cero en el espectro para evitar log(0) = -∞.
        
        En el cálculo de dB: dB = 10*log10(|X[k]|²)
        Si |X[k]| = 0, entonces log10(0) = -∞
        
        Args:
            datos: Array numpy con valores del espectro
            
        Returns:
            Array con ceros reemplazados por el mínimo valor no-cero
        """
        min_nonzero = np.min(datos[np.nonzero(datos)])
        datos[datos == 0] = min_nonzero
        return datos

    def _huellas_digitales(self, muestras, Fs=FS_DEFAULT, 
                          tamano_ventana=EFFECT_WINDOW_SIZE, 
                          relacion_superposicion=RELATIONSHIP_OVERLAP_BY_DEFECT, 
                          amp_min=-60):
        """
        Genera las huellas digitales del audio mediante análisis espectral.
        
        PROCESO MATEMÁTICO DETALLADO:
        ============================
        
        1. SEGMENTACIÓN TEMPORAL:
           - División del audio en ventanas de tamaño N
           - Superposición del 50% para continuidad
           - hop_size = N * (1 - overlap_ratio)
        
        2. VENTANEO (WINDOWING):
           - Aplicación de ventana de Hanning: w[n] = 0.5(1 - cos(2πn/(N-1)))
           - Reduce efectos de discontinuidad en los bordes
           - Mejora la resolución espectral
        
        3. TRANSFORMADA RÁPIDA DE FOURIER:
           - X[k] = Σ_{n=0}^{N-1} x[n] * w[n] * e^{-i2πkn/N}
           - Solo se toman las frecuencias positivas (N/2 primeros bins)
           - Cada bin k corresponde a frecuencia: f[k] = k * Fs / N
        
        4. CONVERSIÓN A DECIBELIOS:
           - dB[k] = 10 * log10(|X[k]|²)
           - Escala logarítmica que simula percepción auditiva humana
        
        5. DETECCIÓN DE PICOS:
           - Filtro de máximos locales en vecindario 2D
           - Un pico en (f,t) es máximo local si es mayor que todos sus vecinos
           - Estructura binaria 2D para definir vecindario
        
        Args:
            muestras: Señal de audio discreta x[n]
            Fs: Frecuencia de muestreo (Hz)
            tamano_ventana: Tamaño N de cada ventana para FFT
            relacion_superposicion: Fracción de superposición entre ventanas
            amp_min: Amplitud mínima en dB para considerar un pico
            
        Returns:
            Lista de tuplas (frecuencia_bin, tiempo_frame) de los picos detectados
        """
        # Cálculo del salto entre ventanas
        hop_size = int(tamano_ventana * (1 - RELATIONSHIP_OVERLAP_BY_DEFECT))
        
        # Número total de frames que se pueden extraer
        n_frames = 1 + (len(muestras) - tamano_ventana) // hop_size if len(muestras) >= tamano_ventana else 0
        
        # Array 2D para almacenar el espectrograma
        arr2D = []
        
        # PASO 1-4: Generación del espectrograma
        for i in range(n_frames):
            start = i * hop_size
            end = start + tamano_ventana
            frame = muestras[start:end]
            
            if len(frame) < tamano_ventana:
                continue
            
            # Aplicación de ventana de Hanning
            # w[n] = 0.5 * (1 - cos(2πn/(N-1)))
            windowed = frame * np.hanning(tamano_ventana)
            
            # Transformada rápida de Fourier
            # Solo frecuencias positivas (simetría hermítica)
            spectrum = np.abs(fft(windowed))[:tamano_ventana // 2]
            
            # Conversión a escala logarítmica (dB)
            # dB = 10 * log10(|X[k]|²) = 20 * log10(|X[k]|)
            spectrum = 10 * np.log10(self._reemplazarCeros(spectrum))
            
            arr2D.append(spectrum)
        
        # Transposición: [freq_bins, time_frames]
        # Cada fila = evolución temporal de una frecuencia
        # Cada columna = espectro instantáneo en un tiempo
        arr2D = np.array(arr2D).T
        
        # Reemplazar -∞ por 0 para estabilidad numérica
        arr2D[arr2D == -np.inf] = 0
        
        # Información de depuración
        print(f"Espectrograma (FFT manual): min={np.min(arr2D)}, max={np.max(arr2D)}")
        print(f"Percentil 90 del espectrograma: {np.percentile(arr2D, 90)}")
        
        # Visualización del espectrograma (solo una vez)
        if not self._ya_graficado and arr2D.size > 0:
            plt.figure(figsize=(10, 4))
            plt.imshow(arr2D, aspect='auto', origin='lower')
            plt.title('Espectrograma (FFT manual, dB)')
            plt.colorbar()
            plt.show()
            self._ya_graficado = True
        
        # PASO 5: DETECCIÓN DE PICOS ESPECTRALES
        # ======================================
        
        # Estructura de vecindario para detección de picos
        # Genera una matriz binaria que define qué píxeles son "vecinos"
        neighborhood = iterate_structure(
            generate_binary_structure(2, 1),  # Conectividad 2D básica (4-conectada)
            AMP_MIN_DEFECT           # Expansión del vecindario
        )
        
        # Filtro de máximos: cada punto se compara con su vecindario
        # maxima[i,j] = True si arr2D[i,j] es el máximo en su vecindario
        maxima = maximum_filter(arr2D, footprint=neighborhood) == arr2D
        
        # Eliminar picos en regiones de silencio (background = 0)
        background = (arr2D == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        
        # Picos válidos: máximos locales que NO están en regiones de silencio
        detected_peaks = maxima & ~eroded_background
        
        # Extracción de coordenadas y amplitudes de los picos
        amps = arr2D[detected_peaks]
        j, i = np.where(detected_peaks)  # j=freq_bin, i=time_frame
        amps = amps.flatten()
        
        # Lista de picos: (time_frame, freq_bin, amplitude)
        peaks = list(zip(i, j, amps))
        
        # Filtrado por amplitud mínima
        peaks_filtered = [x for x in peaks if x[2] > amp_min]
        
        # Extracción de coordenadas para generación de hashes
        frecuencia_picos = [x[1] for x in peaks_filtered]  # Bins de frecuencia
        tiempo_picos = [x[0] for x in peaks_filtered]      # Frames de tiempo
        
        return list(zip(frecuencia_picos, tiempo_picos))

    def _generar_hashes(self, picos, valor_fan=DEFAULT_FAN_VALUE):
        """
        Genera hashes basados en constelaciones de picos (algoritmo tipo Shazam).
        
        ALGORITMO DE CONSTELACIONES:
        ===========================
        
        El concepto clave de Shazam es crear "constelaciones" de picos que sean:
        1. Invariantes al ruido de fondo
        2. Robustas ante distorsiones temporales menores
        3. Únicas para cada canción
        
        PROCESO:
        1. Para cada pico de referencia en tiempo t1 y frecuencia f1
        2. Considerar los próximos 'valor_fan' picos cronológicamente
        3. Para cada par (pico_referencia, pico_objetivo):
           - f1, f2: frecuencias de los picos
           - t1, t2: tiempos de los picos
           - Δt = t2 - t1: diferencia temporal
        4. Generar hash: H = SHA1(f1|f2|Δt)
        5. Almacenar tupla: (hash, t1) - el hash con su tiempo de referencia
        
        VENTAJAS DE ESTE MÉTODO:
        - Los hashes son invariantes ante cambios de volumen
        - Robustos ante ruido que no afecte los picos principales
        - La diferencia temporal Δt hace el hash más específico
        - El tiempo de referencia t1 permite alineación temporal
        
        Args:
            picos: Lista de tuplas (frecuencia, tiempo) de picos detectados
            valor_fan: Número de picos futuros a considerar por cada pico de referencia
            
        Yields:
            Tuplas (hash_string, tiempo_referencia) para cada constelación
        """
        # Ordenamiento cronológico de picos para procesamiento secuencial
        if ORDER_PICES:
            picos.sort(key=lambda x: x[1])  # Ordenar por tiempo (índice 1)
        
        # Generación de constelaciones
        for i in range(len(picos)):
            # Pico de referencia
            freq1 = picos[i][IDX_FREQ_I]  # Frecuencia del pico de referencia
            t1 = picos[i][IDX_TIME_J]     # Tiempo del pico de referencia
            
            # Considerar los próximos 'valor_fan' picos
            for j in range(1, valor_fan):
                if (i + j) < len(picos):
                    # Pico objetivo (futuro)
                    freq2 = picos[i + j][IDX_FREQ_I]  # Frecuencia del pico objetivo
                    t2 = picos[i + j][IDX_TIME_J]     # Tiempo del pico objetivo
                    
                    # Diferencia temporal entre picos
                    t_delta = t2 - t1
                    
                    # Filtro temporal: solo considerar picos dentro del rango válido
                    if MIN_DELTA_TIME_HASH <= t_delta <= MAX_DELTA_TIME_HASH:
                        # Generación del hash de la constelación
                        # Formato: "freq1|freq2|delta_tiempo"
                        constellation_string = f"{freq1}|{freq2}|{t_delta}"
                        
                        # Hash criptográfico SHA-1 para unicidad
                        h = hashlib.sha1(constellation_string.encode('utf-8'))
                        
                        # Truncamiento del hash para eficiencia de almacenamiento
                        hash_truncated = h.hexdigest()[0:FOOTPRINT_REDUCTION]
                        
                        # Yield de la tupla (hash, tiempo_referencia)
                        yield (hash_truncated, t1)

    def _extraer_hashes_audio(self, file_path):
        """
        Extrae todos los hashes de un archivo de audio.
        
        PIPELINE COMPLETO:
        =================
        1. Carga del audio con librosa
        2. Normalización de amplitud
        3. Generación de huellas digitales (picos espectrales)
        4. Conversión de picos a hashes de constelaciones
        
        Args:
            file_path: Ruta del archivo de audio
            
        Returns:
            Lista de tuplas (hash, tiempo_referencia)
        """
        # Carga de audio con frecuencia de muestreo estándar
        y, sr = librosa.load(file_path, sr=FS_DEFAULT, mono=True)
        
        print(f"Audio: {file_path} - min={np.min(y)}, max={np.max(y)}")
        
        # Normalización de amplitud para estabilidad numérica
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))  # Normalización al rango [-1, 1]
        
        # Extracción de picos espectrales
        picos = self._huellas_digitales(y, Fs=sr)
        print(f"Archivo: {file_path} - Picos detectados: {len(picos)}")
        
        # Generación de hashes de constelaciones
        hashes = list(self._generar_hashes(picos))
        print(f"Archivo: {file_path} - Hashes generados: {len(hashes)}")
        
        return hashes

    def _comparar_hashes(self, hashes1, hashes2):
        """
        Compara dos conjuntos de hashes para encontrar coincidencias.
        
        ALGORITMO DE COINCIDENCIA TEMPORAL:
        ==================================
        
        El algoritmo de Shazam no solo cuenta coincidencias de hashes,
        sino que busca ALINEACIONES TEMPORALES consistentes.
        
        PROCESO:
        1. Por cada hash coincidente entre las dos canciones
        2. Calcular el desplazamiento temporal: offset = t2 - t1
        3. Contar cuántos hashes tienen el MISMO desplazamiento
        4. El desplazamiento más frecuente indica la mejor alineación
        5. El score es el número de hashes alineados temporalmente
        
        INTERPRETACIÓN:
        - Si dos canciones son iguales, la mayoría de hashes tendrán el mismo offset
        - Si son diferentes, los offsets serán aleatorios (pocas coincidencias por offset)
        - El offset más frecuente indica dónde una canción aparece dentro de la otra
        
        Args:
            hashes1: Lista de (hash, tiempo) de la primera canción
            hashes2: Lista de (hash, tiempo) de la segunda canción
            
        Returns:
            Score de similitud (número máximo de hashes con offset consistente)
        """
        coincidencias = []  # Lista de desplazamientos temporales
        
        # Búsqueda de hashes coincidentes
        for hash1, offset1 in hashes1:
            for hash2, offset2 in hashes2:
                if hash1 == hash2:  # Hash coincidente encontrado
                    # Calcular desplazamiento temporal
                    desplazamiento_temporal = offset2 - offset1
                    coincidencias.append(desplazamiento_temporal)
        
        if not coincidencias:
            return 0  # No hay coincidencias
        
        # Contar frecuencia de cada desplazamiento
        from collections import Counter
        diff_counter = Counter(coincidencias)
        
        # El score es la frecuencia del desplazamiento más común
        # Esto indica cuántos hashes están temporalmente alineados
        return max(diff_counter.values())

    def find_similar_audio(self, target_file, threshold=0.1):
        """
        Encuentra archivos de audio similares al archivo objetivo.
        
        ALGORITMO DE BÚSQUEDA:
        =====================
        1. Extraer hashes del archivo objetivo
        2. Para cada archivo en la base de datos:
           a. Extraer sus hashes
           b. Comparar con el objetivo
           c. Calcular score de similitud
        3. Ordenar por score descendente
        4. Retornar mejores coincidencias
        
        Args:
            target_file: Archivo de audio a buscar
            threshold: Umbral mínimo de similitud (no usado actualmente)
            
        Returns:
            Lista de tuplas (archivo, score) ordenada por similitud
        """
        # Extracción de hashes del archivo objetivo
        hashes_target = self._extraer_hashes_audio(target_file)
        mejores = []
        
        # Comparación con todos los archivos en el directorio
        for file_path in self.audio_dir.glob('*.wav'):
            # Extracción de hashes del candidato
            hashes_candidato = self._extraer_hashes_audio(str(file_path))
            
            # Cálculo de similitud
            score = self._comparar_hashes(hashes_target, hashes_candidato)
            print(f"Comparando {target_file} con {file_path}: score={score}")
            
            if score > 0:
                mejores.append((str(file_path), score))
        
        # Ordenamiento por score (mayor similitud primero)
        mejores = sorted(mejores, key=lambda x: x[1], reverse=True)
        
        if mejores:
            return [mejores[0]]  # Retornar solo la mejor coincidencia
        return []

    def find_match_in_all_audios(self, fragment_file, block_duration, min_matches=2):
        """
        Wrapper para mantener compatibilidad con interface existente.
        """
        return self.find_similar_audio(fragment_file)

    def plot_fft_comparison(self, recent_path, match_path):
        """
        Visualiza la comparación espectral entre dos archivos de audio.
        
        VISUALIZACIÓN ESPECTRAL:
        =======================
        Genera espectrogramas de ambos archivos para análisis visual.
        Los espectrogramas muestran la evolución temporal del contenido frecuencial.
        
        INTERPRETACIÓN:
        - Eje X: tiempo
        - Eje Y: frecuencia  
        - Color: intensidad (dB)
        - Patrones similares indican contenido musical similar
        
        Args:
            recent_path: Ruta del primer archivo
            match_path: Ruta del segundo archivo
        """
        import matplotlib.pyplot as plt
        
        # Carga de ambos archivos
        y1, sr1 = librosa.load(recent_path, sr=FS_DEFAULT, mono=True)
        y2, sr2 = librosa.load(match_path, sr=FS_DEFAULT, mono=True)
        
        # Creación de subplots para comparación
        plt.figure(figsize=(14, 6))
        
        # Espectrograma del primer archivo
        plt.subplot(1, 2, 1)
        plt.specgram(y1, Fs=sr1, 
                    NFFT=EFFECT_WINDOW_SIZE, 
                    noverlap=int(EFFECT_WINDOW_SIZE * RELATIONSHIP_OVERLAP_BY_DEFECT))
        plt.title(f"Espectrograma: {os.path.basename(recent_path)}")
        plt.xlabel("Tiempo")
        plt.ylabel("Frecuencia")
        plt.colorbar().set_label('Intensidad (dB)')
        
        # Espectrograma del segundo archivo
        plt.subplot(1, 2, 2)
        plt.specgram(y2, Fs=sr2, 
                    NFFT=EFFECT_WINDOW_SIZE, 
                    noverlap=int(EFFECT_WINDOW_SIZE * RELATIONSHIP_OVERLAP_BY_DEFECT))
        plt.title(f"Espectrograma: {os.path.basename(match_path)}")
        plt.xlabel("Tiempo")
        plt.ylabel("Frecuencia")
        plt.colorbar().set_label('Intensidad (dB)')
        
        plt.tight_layout()
        plt.show()


"""
RESUMEN DEL ALGORITMO Y SU RELACIÓN CON SHAZAM:
===============================================

1. FUNDAMENTOS MATEMÁTICOS:
   - Transformada de Fourier para análisis frecuencial
   - Teorema de Nyquist para muestreo adecuado
   - Principio de incertidumbre tiempo-frecuencia
   - Escala logarítmica (dB) para percepción auditiva

2. PROCESO TIPO SHAZAM:
   a) Conversión temporal → frecuencial (FFT)
   b) Detección de picos espectrales prominentes
   c) Generación de "constelaciones" de picos
   d) Hashing criptográfico para comparación eficiente
   e) Alineación temporal para detección robusta

3. VENTAJAS DEL ALGORITMO:
   - Robusto ante ruido de fondo
   - Invariante ante cambios de volumen
   - Eficiente computacionalmente
   - Escalable a grandes bases de datos

4. APLICACIONES:
   - Reconocimiento de música (Shazam, SoundHound)
   - Detección de plagio musical
   - Identificación de audio en contenido multimedia
   - Sistemas de recomendación musical

El código implementa los principios fundamentales que hacen funcionar
aplicaciones como Shazam, proporcionando una base sólida para
el reconocimiento automático de audio.
"""