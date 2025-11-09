import numpy as np
from scipy.io.wavfile import write
import os

from pydub import AudioSegment

# Пути к вашим 6 аудиофайлам
filepaths = [
    "sine_1000Hz_10dB_SPL.wav",
    "sine_1000Hz_20dB_SPL.wav",
    "sine_1000Hz_30dB_SPL.wav",
    "sine_1000Hz_40dB_SPL.wav",
    "sine_1000Hz_50dB_SPL.wav",
    "sine_1000Hz_60dB_SPL.wav"
]

combined = AudioSegment.empty()

for filepath in filepaths:
    audio = AudioSegment.from_wav(filepath)
    combined += audio  # добавляем файл в конец

# Сохраняем объединённый файл
combined.export("combined_output.wav", format="wav")

print("Объединённый файл сохранён как combined_output.wav")

# Настройки сигнала
sample_rate = 44100  # Гц
duration = 1.0       # секунды
frequency = 1000     # Гц
spl_levels = range(10, 61, 10)  # 10, 20, ..., 60 дБ SPL

# Папка для файлов
output_dir = "sine_waves_dB_SPL"
os.makedirs(output_dir, exist_ok=True)

# Генерация сигналов
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

for spl in spl_levels:
    # Перевод дБ SPL в относительную амплитуду (предполагаем 60 дБ SPL = 0 dBFS)
    amplitude = 10 ** ((spl - 60) / 20)
    
    # Генерация синусоиды
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Преобразование в int16 для WAV
    signal_int16 = np.int16(signal * 32767)
    
    # Сохранение в WAV-файл
    filename = f"sine_{frequency}Hz_{spl}dB_SPL.wav"
    filepath = os.path.join(output_dir, filename)
    write(filepath, sample_rate, signal_int16)
    print(f"Сохранено: {filepath}")

print(" Файлы сгенерированы.")


