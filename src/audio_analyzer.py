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
            
        # Encontrar los picos más prominentes en el espectro
        peak_indices = self._find_peaks(fft_result)
        
        # Crear un vector de características basado en los picos
        feature_vector = np.zeros(len(fft_result))
        for idx in peak_indices:
            feature_vector[idx] = fft_result[idx]
            
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