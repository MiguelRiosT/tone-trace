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
            signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            signal = librosa.util.normalize(signal)
            return signal, sr
        except Exception as e:
            print(f"Error al cargar el archivo {file_path}: {e}")
            return None, None
    
    def compute_fft(self, signal):
        """Calcula la FFT de una señal de audio"""
        if signal is None:
            return None, None
            
        window = np.hanning(len(signal))
        windowed_signal = signal * window
        
        fft_result = fft(windowed_signal)
        freqs = fftfreq(len(signal), 1/self.sample_rate)
        
        positive_freq_mask = freqs >= 0
        freqs = freqs[positive_freq_mask]
        fft_result = np.abs(fft_result[positive_freq_mask])
        fft_result = fft_result / np.max(fft_result)
        
        return freqs, fft_result
    
    def extract_features(self, signal):
        """Extrae características de la señal usando FFT"""
        if signal is None:
            return None
            
        freqs, fft_result = self.compute_fft(signal)
        if freqs is None:
            return None
            
        peak_indices = self._find_peaks(fft_result)
        feature_vector = np.zeros(len(fft_result))
        for idx in peak_indices:
            feature_vector[idx] = fft_result[idx]
            
        return feature_vector
    
    def _find_peaks(self, signal, threshold=0.1, min_distance=10):
        """Encuentra los picos más prominentes en una señal"""
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if not any(abs(i-p) < min_distance for p in peaks):
                    peaks.append(i)
        return sorted(peaks, key=lambda x: signal[x], reverse=True)[:20]
    
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
        for file_path in self.assets_dir.glob("*.wav"):
            if str(file_path) == target_file:
                continue
                
            similarity = self.compare_audio_files(target_file, str(file_path))
            if similarity >= threshold:
                similar_files.append((str(file_path), similarity))
        
        similar_files = sorted(similar_files, key=lambda x: x[1], reverse=True)

        if similar_files:
            similar_file, similarity = similar_files[0]
            print(f"Archivo más similar: {similar_file} (Similitud: {similarity:.2f})")
            self.plot_comparison_spectrums(target_file, similar_file, similarity)
        
        return similar_files

    def plot_comparison_spectrums(self, recent_file, similar_file, similarity_score):
        """Muestra dos espectros de frecuencia comparando un archivo reciente con uno similar"""
        signal1, _ = self.load_audio(recent_file)
        signal2, _ = self.load_audio(similar_file)
        
        if signal1 is None or signal2 is None:
            print("No se pudieron cargar ambos archivos para comparar.")
            return
        
        freqs1, fft1 = self.compute_fft(signal1)
        freqs2, fft2 = self.compute_fft(signal2)
        
        if freqs1 is None or freqs2 is None:
            print("No se pudieron calcular las FFT.")
            return
        
        plt.figure(figsize=(14, 6))
        plt.suptitle(f"Comparación de espectros (Similitud: {similarity_score:.2f})", fontsize=16)
        
        plt.subplot(1, 2, 1)
        plt.plot(freqs1, fft1, color='blue')
        plt.title(f"Archivo reciente: {os.path.basename(recent_file)}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(freqs2, fft2, color='orange')
        plt.title(f"Coincidencia: {os.path.basename(similar_file)}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
