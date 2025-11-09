import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from kan import MultKAN
import torch.nn.functional as F
from scipy.interpolate import interp1d

# -----------------------------
# 1. Пути к папкам с данными
# -----------------------------
directories = {
    "воздух": {"train": r"/content/drive/MyDrive/KAN/Data_3/Murs_Air_123/Murs_Air_50_edu",
               "test":  r"/content/drive/MyDrive/KAN/Data_3/Murs_Air_123/Murs_Air_25_test",
               "val":   r"/content/drive/MyDrive/KAN/Data_3/Murs_Air_123/Murs_Air_25_val"},
    "вода":  {"train": r"/content/drive/MyDrive/KAN/Data_3/Murs_Eau_124/Murs_Eau_50_edu",
               "test":  r"/content/drive/MyDrive/KAN/Data_3/Murs_Eau_124/Murs_Eau_25_test",
               "val":   r"/content/drive/MyDrive/KAN/Data_3/Murs_Eau_124/Murs_Eau_25_val"},
    "сок":   {"train": r"/content/drive/MyDrive/KAN/Data_3/Murs_Jus_125/Murs_Jus_50_edu",
               "test":  r"/content/drive/MyDrive/KAN/Data_3/Murs_Jus_125/Murs_Jus_25_test",
               "val":   r"/content/drive/MyDrive/KAN/Data_3/Murs_Jus_125/Murs_Jus_25_val"},
    "молоко":{"train": r"/content/drive/MyDrive/KAN/Data_3/Murs_Lait_125_jour_1/Murs_Lait_50_edu_jour_1",
               "test":  r"/content/drive/MyDrive/KAN/Data_3/Murs_Lait_125_jour_1/Murs_Lait_25_test_jour_1",
               "val":   r"/content/drive/MyDrive/KAN/Data_3/Murs_Lait_125_jour_1/Murs_Lait_25_val_jour_1"},
    "квас":  {"train": r"/content/drive/MyDrive/KAN/Data_3/Murs_Kvas_125/Murs_Kvas_50_edu",
               "test":  r"/content/drive/MyDrive/KAN/Data_3/Murs_Kvas_125/Murs_Kvas_25_test",
               "val":   r"/content/drive/MyDrive/KAN/Data_3/Murs_Kvas_125/Murs_Kvas_25_val"},
    "масло": {"train": r"/content/drive/MyDrive/KAN/Data_3/Murs_Huile_125/Murs_Huile_50_edu",
               "test":  r"/content/drive/MyDrive/KAN/Data_3/Murs_Huile_125/Murs_Huile_25_test",
               "val":   r"/content/drive/MyDrive/KAN/Data_3/Murs_Huile_125/Murs_Huile_25_val"}

}

class_mapping = {"воздух":0, "вода":1, "сок":2, "молоко":3, "квас":4, "масло":5}

# -----------------------------
# 2. Интерполяция сигнала до фиксированной длины
# -----------------------------
def interpolate_signal(signal, target_length=65):
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, signal, kind='linear')
    return f(x_new)

# -----------------------------
# 3. Чтение сигналов из папки
# -----------------------------
def load_signals_from_folder(folder_path, target_length=65):
    signals = []
    file_names = os.listdir(folder_path)
    for f_name in file_names:
        full_path = os.path.join(folder_path, f_name)
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            amplitudes = [item["Amplitude"] for item in data if "Amplitude" in item]
            if len(amplitudes) > 0:
                signals.append(interpolate_signal(amplitudes, target_length))
    return signals

# -----------------------------
# 4. Загружаем все данные
# -----------------------------
target_length = 65

X_train, y_train = [], []
X_test, y_test = [], []
X_val, y_val = [], []

for cls_name, paths in directories.items():
    label = class_mapping[cls_name]
    
    train_signals = load_signals_from_folder(paths["train"], target_length)
    X_train.extend(train_signals)
    y_train.extend([label]*len(train_signals))
    
    test_signals = load_signals_from_folder(paths["test"], target_length)
    X_test.extend(test_signals)
    y_test.extend([label]*len(test_signals))
    
    val_signals = load_signals_from_folder(paths["val"], target_length)
    X_val.extend(val_signals)
    y_val.extend([label]*len(val_signals))

# -----------------------------
# 5. Преобразуем в numpy и нормализуем
# -----------------------------
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int64)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.int64)

def normalize_signals(X):
    return (X - X.min(axis=1, keepdims=True)) / \
           (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8)

X_train = normalize_signals(X_train)
X_test = normalize_signals(X_test)
X_val = normalize_signals(X_val)

# -----------------------------
# 6. Конвертируем в PyTorch тензоры
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# -----------------------------
# 7. Создаём MultKAN (старый формат, create() не нужен)
# -----------------------------
num_classes = len(class_mapping)
width = [target_length, 64, num_classes]  # [вход, скрытые, выход]
model = MultKAN(width=width)  # готовая модель

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 8. Обучение MultKAN
# -----------------------------
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = F.cross_entropy(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).argmax(dim=1)
        val_acc = (val_pred == y_val).float().mean()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.3f}")

# -----------------------------
# 9. Проверка параметров модели
# -----------------------------
print("\nПараметры модели после обучения:")
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

# -----------------------------
# 10. Тестирование
# -----------------------------
model.eval()
with torch.no_grad():
    test_pred = model(X_test).argmax(dim=1)
    test_acc = (test_pred == y_test).float().mean()
print(f"Test Accuracy: {test_acc:.3f}")

# -----------------------------
# 11. Сохранение обученной модели
# -----------------------------
# checkpoint_dir = r"/content/drive/MyDrive/KAN/model_2"
# model_id = "0.1"  # версия модели (можно любую строку)

 # Создаём папку, если её нет
# os.makedirs(checkpoint_dir, exist_ok=True)

 # Сохраняем модель (вес + конфигурация)
# model.save(model_id=model_id, checkpoint_dir=checkpoint_dir)

# print(f"Модель сохранена в папке: {checkpoint_dir}, версия: {model_id}")
