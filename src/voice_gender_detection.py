import numpy as np
import sounddevice as sd

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

    # Usamos solo la mitad positiva
    magnitude = np.abs(audio_fft)[:n//2]
    freq = freq[:n//2]

    return freq, magnitude

def detect_gender(freq, magnitude):
    peak_idx = np.argmax(magnitude)
    peak_freq = freq[peak_idx]

    print(f"Frecuencia pico detectada: {peak_freq:.2f} Hz")

    if 85 <= peak_freq <= 180:
        return "Hombre"
    elif 165 <= peak_freq <= 255:
        return "Mujer"
    else:
        return "Indefinido"
    
def main():
    audio, samplerate = record_audio()
    freq, magnitude = analyze_frequency(audio, samplerate)
    gender = detect_gender(freq, magnitude)
    print(f"Se detectó voz de: {gender}")

if __name__ == "__main__":
    main()