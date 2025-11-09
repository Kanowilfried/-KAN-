import math
import re

# входные данные
filepaths = [
    "sine_1000Hz_10dB_SPL.wav",
    "sine_1000Hz_20dB_SPL.wav",
    "sine_1000Hz_30dB_SPL.wav",
    "sine_1000Hz_40dB_SPL.wav",
    "sine_1000Hz_50dB_SPL.wav",
    "sine_1000Hz_60dB_SPL.wav"
]

# параметры акустики
r0 = 1.0                  # опорное расстояние, м (где измерено L1)
alpha = 0.005             # атм. поглощение, дБ/м (пример)
target_spl = 0.0          # искомый уровень в дБ
frequency = 1000.0        # частота сигнала, Гц

# параметры панели (пластика)
thickness_m = 0.004       # толщина, м (4 мм)
density = 1380.0          # плотность пластика, kg/m^3 (пример для PET)
# можно подправить density если другой пластик

def extract_spl_from_name(name):
    m = re.search(r'_(\d{1,3})dB_SPL', name)
    if not m:
        raise ValueError(f"Не удалось извлечь SPL из имени: {name}")
    return float(m.group(1))

def mass_per_area(density, thickness):
    return density * thickness  # kg/m^2

def mass_law_TL(m_surf, freq):
    # mass law: TL = 20 log10(m * f) - 47   (приближённо)
    # m в кг/м^2, f в Гц, TL в дБ
    val = 20.0 * math.log10(max(m_surf * freq, 1e-12)) - 47.0
    return val

def distance_for_target(L1, L_target, r0=1.0, alpha=0.0):
    # решаем L1 - 20*log10(r/r0) - alpha*(r-r0) = L_target
    def f(r):
        return L1 - 20.0*math.log10(r/r0) - alpha*(r - r0) - L_target

    if f(r0) <= 0:
        return r0

    lo = r0
    hi = r0 * 2.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e7:
            raise RuntimeError("Не удалось найти верхнюю границу для бинпоиска")

    for _ in range(80):
        mid = 0.5*(lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)

# расчёты
m_surf = mass_per_area(density, thickness_m)
TL_panel = mass_law_TL(m_surf, frequency)

print(f"Параметры панели: плотность={density} kg/m^3, толщина={thickness_m*1000:.1f} mm")
print(f"Масса на единицу площади m = {m_surf:.3f} kg/m^2")
print(f"Оценка по mass law: TL @ {frequency:.0f} Hz ≈ {TL_panel:.2f} dB")
print()

results = []
for fp in filepaths:
    L1 = extract_spl_from_name(fp)
    L1_after_panel = L1 - TL_panel  # ослабление при прохождении через панель
    r = distance_for_target(L1_after_panel, target_spl, r0=r0, alpha=alpha)
    results.append((fp, L1, TL_panel, L1_after_panel, r))

for fp, L1, TL, L1p, r in results:
    print(f"{fp:35s} L1={L1:5.1f} dB  TL_panel={TL:6.2f} dB  L_after={L1p:6.2f} dB  -> dist to {target_spl:.1f} dB: {r:.3f} m")
