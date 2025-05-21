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
        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            signal = librosa.util.normalize(signal)
            return signal, sr
        except Exception as e:
            print(f"Error al cargar el archivo {file_path}: {e}")
            return None, None
    
    def compute_fft(self, signal):
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
        if signal is None:
            return None

        freqs, fft_result = self.compute_fft(signal)
        if freqs is None:
            return None

        N = 2048
        fft_result_resized = np.interp(
            np.linspace(0, len(fft_result), N),
            np.arange(len(fft_result)),
            fft_result
        )   

        peak_indices = self._find_peaks(fft_result_resized)
        
        feature_vector = np.zeros(N)
        for idx in peak_indices:
            feature_vector[idx] = fft_result_resized[idx]
            
        return feature_vector
    
    def _find_peaks(self, signal, threshold=0.1, min_distance=10):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if not any(abs(i-p) < min_distance for p in peaks):
                    peaks.append(i)
        return sorted(peaks, key=lambda x: signal[x], reverse=True)[:20]
    
    def analyze_audio(self, file_path):
        signal, _ = self.load_audio(file_path)
        if signal is None:
            return None
            
        return self.extract_features(signal)
    
    def compare_audio_files(self, file1, file2):
        features1 = self.analyze_audio(file1)
        features2 = self.analyze_audio(file2)
        
        if features1 is None or features2 is None:
            return 0.0
            
        similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
        return similarity
    
    def find_similar_audio(self, target_file, threshold=0.5):
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
                
                # Mostrar una vez la comparación gráfica
                self.plot_comparison(target_file, str(file_path))
                break

        return sorted(similar_files, key=lambda x: x[1], reverse=True)
    
    def plot_spectrum(self, file_path, save_path=None):
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

            threshold_dynamic = np.percentile(all_scores, 90)
            strong_matches = [s for s in all_scores if s >= threshold_dynamic]

            if len(strong_matches) >= min_matches:
                matches.append((str(file_path), max(strong_matches)))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def plot_fft_comparison(self, recent_path, match_path):
        """Grafica los espectros de frecuencia (FFT) del audio reciente y el coincidente"""
        signal1, _ = self.load_audio(recent_path)
        signal2, _ = self.load_audio(match_path)

        if signal1 is None or signal2 is None:
            print("Error cargando alguno de los audios para comparación.")
            return

        freqs1, fft1 = self.compute_fft(signal1)
        freqs2, fft2 = self.compute_fft(signal2)

        if freqs1 is None or freqs2 is None:
            print("Error al calcular la FFT.")
            return

        plt.figure(figsize=(14, 6))
        plt.suptitle(f"Comparación de espectros de frecuencia", fontsize=16)

        plt.subplot(1, 2, 1)
        plt.plot(freqs1, fft1, color='blue')
        plt.title(f"Espectro: {os.path.basename(recent_path)}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(freqs2, fft2, color='orange')
        plt.title(f"Espectro: {os.path.basename(match_path)}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()