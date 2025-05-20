import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
from pathlib import Path
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity

class AudioAnalyzer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.assets_dir = Path("assets/audio")
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
    def load_audio(self, file_path):
        """Carga un archivo de audio y retorna la señal y la frecuencia de muestreo"""
        try:
            # Cargar el audio y asegurar que esté en mono
            signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Normalizar la señal
            signal = librosa.util.normalize(signal)
            
            return signal, sr
        except Exception as e:
            print(f"Error al cargar el archivo {file_path}: {e}")
            return None, None
    
    def compute_fft(self, signal):
        """Calcula la FFT de una señal de audio"""
        if signal is None:
            return None, None
            
        # Aplicar ventana de Hann para reducir el efecto de los bordes
        window = np.hanning(len(signal))
        windowed_signal = signal * window
        
        # Calcular FFT
        fft_result = fft(windowed_signal)
        
        # Calcular frecuencias correspondientes
        freqs = fftfreq(len(signal), 1/self.sample_rate)
        
        # Solo tomar la parte positiva de la FFT (frecuencias positivas)
        positive_freq_mask = freqs >= 0
        freqs = freqs[positive_freq_mask]
        fft_result = np.abs(fft_result[positive_freq_mask])
        
        # Normalizar la FFT
        fft_result = fft_result / np.max(fft_result)
        
        return freqs, fft_result
    
    def extract_features(self, signal):
        """Extrae características de la señal usando FFT"""
        if signal is None:
            return None

        freqs, fft_result = self.compute_fft(signal)
        if freqs is None:
            return None

        # Ajustar a longitud fija
        N = 2048  # Puedes cambiar este tamaño si lo deseas
        #N = 4096 
        fft_result_resized = np.interp(
            np.linspace(0, len(fft_result), N),
            np.arange(len(fft_result)),
            fft_result
        )   

        # Encontrar los picos más prominentes en el espectro reescalado
        peak_indices = self._find_peaks(fft_result_resized)
        
        # Crear un vector de características basado en los picos
        feature_vector = np.zeros(N)
        for idx in peak_indices:
            feature_vector[idx] = fft_result_resized[idx]
            
        return feature_vector
    
    def _find_peaks(self, signal, threshold=0.1, min_distance=10):
        """Encuentra los picos más prominentes en una señal"""
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # Verificar que no esté muy cerca de otro pico
                if not any(abs(i-p) < min_distance for p in peaks):
                    peaks.append(i)
        return sorted(peaks, key=lambda x: signal[x], reverse=True)[:20]  # Retornar los 20 picos más fuertes
    
    def analyze_audio(self, file_path):
        """Analiza un archivo de audio y retorna sus características"""
        signal, _ = self.load_audio(file_path)
        if signal is None:
            return None
            
        return self.extract_features(signal)
    
    def compare_audio_files(self, file1, file2):
        """Compara dos archivos de audio y retorna su similitud"""
        features1 = self.analyze_audio(file1)
        features2 = self.analyze_audio(file2)
        
        if features1 is None or features2 is None:
            return 0.0
            
        # Calcular similitud del coseno
        similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
        return similarity
    
    def find_similar_audio(self, target_file, threshold=0.5):
        """Encuentra archivos de audio similares al archivo objetivo"""
        if not os.path.exists(target_file):
            print(f"El archivo {target_file} no existe")
            return []
            
        target_features = self.analyze_audio(target_file)
        if target_features is None:
            return []
            
        similar_files = []
        
        # Analizar todos los archivos en la carpeta de assets
        for file_path in self.assets_dir.glob("*.wav"):
            if str(file_path) == target_file:
                continue
                
            similarity = self.compare_audio_files(target_file, str(file_path))
            if similarity >= threshold:
                similar_files.append((str(file_path), similarity))
                
        # Ordenar por similitud (de mayor a menor)
        return sorted(similar_files, key=lambda x: x[1], reverse=True)
    
    def plot_spectrum(self, file_path, save_path=None):
        """Genera y muestra/guarda un gráfico del espectro de frecuencia"""
        signal, _ = self.load_audio(file_path)
        if signal is None:
            return
            
        freqs, fft_result = self.compute_fft(signal)
        if freqs is None:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(freqs, np.abs(fft_result))
        plt.title(f"Espectro de Frecuencia: {os.path.basename(file_path)}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 
       
    def find_match_in_all_audios(self, fragment_file, block_duration, min_matches=2):
        """
        Encuentra audios similares al fragmento, usando:
        - comparación por bloques con overlap
        - threshold dinámico por archivo
        - al menos 'min_matches' bloques por archivo deben coincidir
        """
        matches = []
        fragment_signal, _ = self.load_audio(fragment_file)
        if fragment_signal is None:
            return matches

        fragment_vector = self.extract_features(fragment_signal)
        if fragment_vector is None:
            return matches

        block_size = int(block_duration * self.sample_rate)

        for file_path in self.assets_dir.glob("*.wav"):
            if str(file_path.resolve()) == str(Path(fragment_file).resolve()):
                continue

            long_signal, _ = self.load_audio(file_path)
            if long_signal is None or len(long_signal) < block_size:
                continue

            all_scores = []

            for i in range(0, len(long_signal) - block_size + 1, block_size // 2):
                block = long_signal[i:i + block_size]
                block_vector = self.extract_features(block)
                if block_vector is None:
                    continue

                sim = cosine_similarity(fragment_vector.reshape(1, -1), block_vector.reshape(1, -1))[0][0]
                all_scores.append(sim)

            if len(all_scores) == 0:
                continue

            # --- Estrategia Combinada ---
            threshold_dynamic = np.percentile(all_scores, 90)  # top 10%
            strong_matches = [s for s in all_scores if s >= threshold_dynamic]

            if len(strong_matches) >= min_matches:
                matches.append((str(file_path), max(strong_matches)))  # guardar la similitud más alta

        return sorted(matches, key=lambda x: x[1], reverse=True)