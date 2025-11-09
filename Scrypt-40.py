import subprocess
import time

# Путь к программе, которую нужно запускать
program_path = "python3 data_play4.py"  # Замените на нужный путь

# Запустить программу 5 раз с интервалом в 5 секунд
for i in range(125):
    print(f"Запуск {i+1}/125")
    subprocess.run(program_path, shell=True)
    if i < 124:  # Ждать только перед следующими 4 запусками
        time.sleep(5)