import os
import numpy as np
import hashlib
from pathlib import Path
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
import matplotlib.pyplot as plt
import librosa
from numpy.fft import fft, fftfreq

AUDIO_DIR = 'assets/audio'
IDX_FREQ_I = 0
IDX_TIME_J = 1
FS_POR_DEFECTO = 44100
TAMANO_VENTANA_POR_DEFECTO = 1024
RELACION_SUPERPOSICION_POR_DEFECTO = 0.5
VALOR_FAN_POR_DEFECTO = 15
AMP_MIN_POR_DEFECTO = 1
TAMANO_VECINDARIO_PICOS = 20
MIN_DELTA_TIEMPO_HASH = 0
MAX_DELTA_TIEMPO_HASH = 200
ORDENAR_PICOS = True
REDUCCION_HUELLA = 20

class AudioAnalyzer:
    def __init__(self, audio_dir=AUDIO_DIR):
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._ya_graficado = False

    def _reemplazarCeros(self, datos):
        min_nonzero = np.min(datos[np.nonzero(datos)])
        datos[datos == 0] = min_nonzero
        return datos

    def _huellas_digitales(self, muestras, Fs=FS_POR_DEFECTO, tamano_ventana=TAMANO_VENTANA_POR_DEFECTO, relacion_superposicion=RELACION_SUPERPOSICION_POR_DEFECTO, amp_min=-60):
        hop_size = int(tamano_ventana * (1 - RELACION_SUPERPOSICION_POR_DEFECTO))
        n_frames = 1 + (len(muestras) - tamano_ventana) // hop_size if len(muestras) >= tamano_ventana else 0
        arr2D = []
        for i in range(n_frames):
            start = i * hop_size
            end = start + tamano_ventana
            frame = muestras[start:end]
            if len(frame) < tamano_ventana:
                continue
            windowed = frame * np.hanning(tamano_ventana)
            spectrum = np.abs(fft(windowed))[:tamano_ventana // 2]
            spectrum = 10 * np.log10(self._reemplazarCeros(spectrum))
            arr2D.append(spectrum)
        arr2D = np.array(arr2D).T  # shape: [freq_bins, time_frames]
        arr2D[arr2D == -np.inf] = 0
        print(f"Espectrograma (FFT manual): min={np.min(arr2D)}, max={np.max(arr2D)}")
        print(f"Percentil 90 del espectrograma: {np.percentile(arr2D, 90)}")
        if not self._ya_graficado and arr2D.size > 0:
            plt.figure(figsize=(10, 4))
            plt.imshow(arr2D, aspect='auto', origin='lower')
            plt.title('Espectrograma (FFT manual, dB)')
            plt.colorbar()
            plt.show()
            self._ya_graficado = True
        neighborhood = iterate_structure(generate_binary_structure(2, 1), TAMANO_VECINDARIO_PICOS)
        maxima = maximum_filter(arr2D, footprint=neighborhood) == arr2D
        background = (arr2D == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks = maxima & ~eroded_background
        amps = arr2D[detected_peaks]
        j, i = np.where(detected_peaks)
        amps = amps.flatten()
        peaks = list(zip(i, j, amps))
        peaks_filtered = [x for x in peaks if x[2] > amp_min]
        frecuencia_picos = [x[1] for x in peaks_filtered]
        tiempo_picos = [x[0] for x in peaks_filtered]
        return list(zip(frecuencia_picos, tiempo_picos))

    def _generar_hashes(self, picos, valor_fan=VALOR_FAN_POR_DEFECTO):
        if ORDENAR_PICOS:
            picos.sort(key=lambda x: x[1])
        for i in range(len(picos)):
            for j in range(1, valor_fan):
                if (i + j) < len(picos):
                    freq1 = picos[i][IDX_FREQ_I]
                    freq2 = picos[i + j][IDX_FREQ_I]
                    t1 = picos[i][IDX_TIME_J]
                    t2 = picos[i + j][IDX_TIME_J]
                    t_delta = t2 - t1
                    if MIN_DELTA_TIEMPO_HASH <= t_delta <= MAX_DELTA_TIEMPO_HASH:
                        h = hashlib.sha1(f"{freq1}|{freq2}|{t_delta}".encode('utf-8'))
                        yield (h.hexdigest()[0:REDUCCION_HUELLA], t1)

    def _extraer_hashes_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=FS_POR_DEFECTO, mono=True)
        print(f"Audio: {file_path} - min={np.min(y)}, max={np.max(y)}")
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        picos = self._huellas_digitales(y, Fs=sr)
        print(f"Archivo: {file_path} - Picos detectados: {len(picos)}")
        hashes = list(self._generar_hashes(picos))
        print(f"Archivo: {file_path} - Hashes generados: {len(hashes)}")
        return hashes

    def _comparar_hashes(self, hashes1, hashes2):
        coincidencias = []
        for hash1, offset1 in hashes1:
            for hash2, offset2 in hashes2:
                if hash1 == hash2:
                    coincidencias.append(offset2 - offset1)
        if not coincidencias:
            return 0
        # El score es la cantidad máxima de coincidencias alineadas
        from collections import Counter
        diff_counter = Counter(coincidencias)
        return max(diff_counter.values())

    def find_similar_audio(self, target_file, threshold=0.1):
        hashes_target = self._extraer_hashes_audio(target_file)
        mejores = []
        for file_path in self.audio_dir.glob('*.wav'):
            # Permitir comparar consigo mismo para depuración
            # if str(file_path.resolve()) == str(Path(target_file).resolve()):
            #     continue
            hashes_candidato = self._extraer_hashes_audio(str(file_path))
            score = self._comparar_hashes(hashes_target, hashes_candidato)
            print(f"Comparando {target_file} con {file_path}: score={score}")
            if score > 0:
                mejores.append((str(file_path), score))
        mejores = sorted(mejores, key=lambda x: x[1], reverse=True)
        if mejores:
            return [mejores[0]]
        return []

    def find_match_in_all_audios(self, fragment_file, block_duration, min_matches=2):
        return self.find_similar_audio(fragment_file)

    def plot_fft_comparison(self, recent_path, match_path):
        import matplotlib.pyplot as plt
        y1, sr1 = librosa.load(recent_path, sr=FS_POR_DEFECTO, mono=True)
        y2, sr2 = librosa.load(match_path, sr=FS_POR_DEFECTO, mono=True)
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.specgram(y1, Fs=sr1, NFFT=TAMANO_VENTANA_POR_DEFECTO, noverlap=int(TAMANO_VENTANA_POR_DEFECTO * RELACION_SUPERPOSICION_POR_DEFECTO))
        plt.title(f"Espectrograma: {os.path.basename(recent_path)}")
        plt.xlabel("Tiempo")
        plt.ylabel("Frecuencia")
        plt.colorbar().set_label('Intensidad (dB)')
        plt.subplot(1, 2, 2)
        plt.specgram(y2, Fs=sr2, NFFT=TAMANO_VENTANA_POR_DEFECTO, noverlap=int(TAMANO_VENTANA_POR_DEFECTO * RELACION_SUPERPOSICION_POR_DEFECTO))
        plt.title(f"Espectrograma: {os.path.basename(match_path)}")
        plt.xlabel("Tiempo")
        plt.ylabel("Frecuencia")
        plt.colorbar().set_label('Intensidad (dB)')
        plt.tight_layout()
        plt.show()