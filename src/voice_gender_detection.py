import numpy as np
import sounddevice as sd

def is_audio_loud_enough(audio, threshold=0.01):
    energy = np.sum(audio ** 2) / len(audio)
    return energy > threshold

def record_audio(duration=3, samplerate=44100):
    print("Grabando...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("Grabación finalizada.")
    return audio.flatten(), samplerate

def analyze_frequency(audio, samplerate):
    n = len(audio)
    audio_fft = np.fft.fft(audio)
    freq = np.fft.fftfreq(n, d=1/samplerate)

    # Solo la mitad positiva
    magnitude = np.abs(audio_fft)[:n//2]
    freq = freq[:n//2]

    return freq, magnitude

def detect_gender(freq, magnitude):
    # Filtrar frecuencias relevantes (85 Hz a 300 Hz)
    mask = (freq >= 85) & (freq <= 300)
    filtered_freq = freq[mask]
    filtered_magnitude = magnitude[mask]

    if len(filtered_magnitude) == 0:
        print("No se detectaron frecuencias relevantes.")
        return "Indefinido"

    peak_idx = np.argmax(filtered_magnitude)
    peak_freq = filtered_freq[peak_idx]

    print(f"Frecuencia pico detectada (filtrada): {peak_freq:.2f} Hz")

    if 85 <= peak_freq <= 180:
        return "Hombre"
    elif 165 <= peak_freq <= 255:
        return "Mujer"
    else:
        return "Indefinido"
    
def main():
    audio, samplerate = record_audio()

    if not is_audio_loud_enough(audio):
        print("No se detectó suficiente sonido para analizar.")
        return
    
    freq, magnitude = analyze_frequency(audio, samplerate)
    gender = detect_gender(freq, magnitude)
    print(f"Se detectó voz de: {gender}")

if __name__ == "__main__":
    main()