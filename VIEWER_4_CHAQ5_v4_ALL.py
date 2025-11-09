import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os

# === Пути к каталогам ===
directory_blue = r'D:\PROGRAMS\KAN\Data_1\Murs_Air_123'
# directory_green = r'D:\PROGRAMS\KAN\Data_1\Air_120' # Нет
directory_red = r'D:\PROGRAMS\KAN\Data_1\Murs_Eau_124'
directory_brown = r'D:\PROGRAMS\KAN\Data_1\Murs_Kvas_125'
directory_orange = r'D:\PROGRAMS\KAN\Data_1\Murs_Jus_125'
directory_violet = r'D:\PROGRAMS\KAN\Data_1\Murs_Lait_125_jour_1'
directory_yellow = r'D:\PROGRAMS\KAN\Data_1\Murs_Lait_124_jour_3'
directory_green = r'D:\PROGRAMS\KAN\Data_1\Murs_Lait_125_jour_4'
directory_pink = r'D:\PROGRAMS\KAN\Data_1\Murs_Lait_125_jour_5'
directory_black = r'D:\PROGRAMS\KAN\Data_1\Murs_Lait_131_jour_6'
directory_grey = r'D:\PROGRAMS\KAN\Data_1\Murs_Huile_125'

# === Параметры сглаживания и времени ===
use_savgol = True
window_size = 11
poly_order = 2
duration_seconds = 6  # Общее время измерений

# === Функция обработки каталога ===
def process_directory(directory, color, source_label):
    file_names = os.listdir(directory)
    file_paths = [os.path.join(directory, f) for f in file_names if os.path.isfile(os.path.join(directory, f))]

    first = True  # флаг для одной подписи в легенде

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        amplitudes = [item["Amplitude"] for item in data if "Amplitude" in item]
        num_measurements = len(amplitudes)
        # print("amplitudes", amplitudes)

        if num_measurements == 0:
            continue  # Пропустить пустые файлы

        time_values = np.linspace(0, duration_seconds, num_measurements, endpoint=False)

        if use_savgol and num_measurements >= window_size:
            amplitudes_smoothed = savgol_filter(amplitudes, window_length=window_size, polyorder=poly_order)
        elif not use_savgol:
            kernel = np.ones(window_size) / window_size
            amplitudes_smoothed = np.convolve(amplitudes, kernel, mode='same')
        else:
            amplitudes_smoothed = amplitudes

        # Устанавливаем подпись только для первого графика из каталога
        label = source_label if first else None
        first = False

        plt.plot(time_values, amplitudes_smoothed, label=label, color=color)


# === Построение графиков для двух каталогов ===
# process_directory(directory_blue, color='blue', source_label='Воздух без преград')
# process_directory(directory_green, color='green', source_label='Воздух со стенами')
# process_directory(directory_red, color='red', source_label='Вода со стенами')

# === Построение графика для всех каталогов ===
process_directory(directory_blue, color='blue', source_label='Воздух со стенами')
process_directory(directory_red, color='red', source_label='Вода')
process_directory(directory_brown, color='brown', source_label='Квас')
process_directory(directory_orange, color='orange', source_label='Сок')
process_directory(directory_violet, color='violet', source_label='Молоко 1й день')
process_directory(directory_yellow, color='yellow', source_label='Молоко 3й день')
process_directory(directory_green, color='pink', source_label='Молоко 4й день')
process_directory(directory_pink, color='pink', source_label='Молоко 4й день')
process_directory(directory_black, color='black', source_label='Молоко 6й день')
process_directory(directory_grey, color='grey', source_label='Масло')


# === Настройки графика ===
plt.title('График амплитуды от времени (сглаженный)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()
plt.show()
