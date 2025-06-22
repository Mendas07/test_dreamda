import csv
import os
from datetime import datetime

# Simulação de resultados já obtidos (exemplo)
DATASET_NAME = "shenzhen"
SEEDS = [0, 1, 2]
ACCURACIES = [0.8234, 0.8461, 0.8389]  # substitua pelos reais

# Garante que a pasta de resultados exista
os.makedirs("results", exist_ok=True)
results_path = f"results/{DATASET_NAME.lower()}.csv"

# Escreve resultados no CSV
with open(results_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if os.stat(results_path).st_size == 0:
        writer.writerow(["timestamp", "dataset", "mode", "seed", "accuracy"])

    for seed, acc in zip(SEEDS, ACCURACIES):
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            DATASET_NAME,
            "pretrained",
            seed,
            round(acc, 4)
        ])

print(f"✅ Resultados salvos em {results_path}")
