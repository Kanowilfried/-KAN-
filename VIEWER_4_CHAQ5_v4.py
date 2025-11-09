import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Конфигурация каталогов и меток ===
DATA_SOURCES = {
    "Воздух со стенами":  ("D:/PROGRAMS/KAN/Data_1/Murs_Air_123", "blue"),
    "Вода":               ("D:/PROGRAMS/KAN/Data_1/Murs_Eau_124", "red"),
    "Квас":               ("D:/PROGRAMS/KAN/Data_1/Murs_Kvas_125", "brown"),
    "Сок":                ("D:/PROGRAMS/KAN/Data_1/Murs_Jus_125", "orange"),
    "Молоко 1й день":     ("D:/PROGRAMS/KAN/Data_1/Murs_Lait_125_jour_1", "violet"),
    "Молоко 3й день":     ("D:/PROGRAMS/KAN/Data_1/Murs_Lait_124_jour_3", "yellow"),
    "Молоко 4й день":     ("D:/PROGRAMS/KAN/Data_1/Murs_Lait_125_jour_4", "green"),
    "Молоко 5й день":     ("D:/PROGRAMS/KAN/Data_1/Murs_Lait_125_jour_5", "pink"),
    "Молоко 6й день":     ("D:/PROGRAMS/KAN/Data_1/Murs_Lait_131_jour_6", "black"),
    "Масло":              ("D:/PROGRAMS/KAN/Data_1/Murs_Huile_125", "grey"),
}

# === Параметры анализа ===
USE_SAVGOL = True
WINDOW_SIZE = 11
POLY_ORDER = 2
DURATION_SECONDS = 6
SAVE_PLOT = True  # если True — сохраняет график в файл

def process_directory(directory: str) -> np.ndarray | None:
    """Читает и усредняет все JSON-файлы в каталоге."""
    if not os.path.exists(directory):
        print(f"❌ Каталог не найден: {directory}")
        return None

    file_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json") and os.path.isfile(os.path.join(directory, f))
    ]
    if not file_paths:
        print(f"⚠️ Нет JSON-файлов в каталоге: {directory}")
        return None

    all_amplitudes = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            amplitudes = [item["Amplitude"] for item in data if "Amplitude" in item]
            if not amplitudes:
                continue
            all_amplitudes.append(amplitudes)
        except Exception as e:
            print(f"⚠️ Ошибка при чтении {path}: {e}")

    if not all_amplitudes:
        print(f"⚠️ В каталоге {directory} нет валидных данных.")
        return None

    # Приведение к общей длине
    min_len = min(map(len, all_amplitudes))
    all_amplitudes = np.array([a[:min_len] for a in all_amplitudes])

    # Усреднение
    mean_amplitude = np.mean(all_amplitudes, axis=0)
    return mean_amplitude


def plot_data():
    plt.figure(figsize=(10, 6))
    for label, (directory, color) in DATA_SOURCES.items():
        mean_amplitude = process_directory(directory)
        if mean_amplitude is None:
            continue

        time_values = np.linspace(0, DURATION_SECONDS, len(mean_amplitude), endpoint=False)

        # Сглаживание
        if USE_SAVGOL and len(mean_amplitude) >= WINDOW_SIZE:
            mean_amplitude = savgol_filter(mean_amplitude, WINDOW_SIZE, POLY_ORDER)
        elif not USE_SAVGOL and len(mean_amplitude) >= WINDOW_SIZE:
            kernel = np.ones(WINDOW_SIZE) / WINDOW_SIZE
            mean_amplitude = np.convolve(mean_amplitude, kernel, mode="same")

        # Построение линии
        plt.plot(time_values, mean_amplitude, color=color, label=label, linewidth=2.5)

    plt.title("Усреднённые графики амплитуды от времени (сглаженные)", fontsize=13)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOT:
        plt.savefig("amplitude_plot.png", dpi=300)
        print("💾 График сохранён как amplitude_plot.png")

    plt.show()


if __name__ == "__main__":
    plot_data()
