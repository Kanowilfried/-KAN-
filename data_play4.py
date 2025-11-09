import serial
import json
import time
import os
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import re
import numpy as np
import sys

file_save_name = 'l_huile'

# ========== CONFIG ==========
serial_port = '/dev/ttyUSB0'
baud_rate = 9600
audio_file = os.path.expanduser('/home/orangepi/workspace/mon_server/mus4_6sec.mp3')
# =============================

# Initialisation des données
current_data = []

# Connexion série
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Attente que l'Arduino démarre

# Lecture des données pendant que la musique joue
def collect_data():
    collected = []
    regression = None

    while arduino.in_waiting:
        line = arduino.readline().decode('utf-8').strip()
        if line:
            print(f"Reçu: {line}")
            try:
                data = json.loads(line)
                collected.append(data)
                current_data.append(data)

                if "Regression" in data:
                    regression = data["Regression"]

            except json.JSONDecodeError:
                print(f"Erreur JSON: {line}")
    return collected, regression

# Lecture de l'audio et capture en parallèle
def main():
    timestamp_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_file = f"data_{timestamp_filename}.json"

    if not os.path.exists(audio_file):
        print(f"Fichier introuvable: {audio_file}")
        return

    audio = AudioSegment.from_mp3(audio_file)
    normalized_audio = audio -30


    print("Lecture audio en cours...")
    start_time = time.time()
    play(normalized_audio)
    duration = time.time() - start_time
    print("🎵 Fin de la lecture audio.")

    # Dernière collecte après la lecture
    collected_data, regression = collect_data()

    # Ajout d'une entrée JSON avec l'heure actuelle
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "timestamp": timestamp,
        "status": "Lecture audio terminée",
    }
    if regression:
        summary["fonction"] = regression

    current_data.append(summary)

    # Sauvegarde du fichier JSON'
    with open(file_save_name + '/' + json_file, 'w') as f:
        json.dump(current_data, f, indent=2)
    print(f"✅ Données enregistrées dans {json_file}")

    # Génération du graphe si fonction présente
    # if regression:
    #     match = re.match(r"f\(x\) = ([\d\.\-eE]+)x \+ ([\d\.\-eE]+)", regression)
    #     if match:
    #         a = float(match.group(1))
    #         b = float(match.group(2))

    #         x_vals = np.linspace(0, 10, 100)
    #         y_vals = a * x_vals + b

    #         plt.plot(x_vals, y_vals, label=regression, color='blue')
    #         plt.title("Graphe de la fonction f(x)")
    #         plt.xlabel("x")
    #         plt.ylabel("f(x)")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig(f"fonction_graph_{timestamp_filename}.png")
    #         print(f"📈 Graphe enregistré sous 'fonction_graph_{timestamp_filename}.png'")
    #     else:
    #         print("⚠️ Format de la fonction non reconnu.")
    # else:
    #     print("ℹ️ Aucune fonction f(x) détectée.")

    sys.exit(0)

# Lancement
main()
