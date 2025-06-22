import pandas as pd
import matplotlib.pyplot as plt

# Caminho para o CSV
CSV_PATH = "results/pets.csv"

# Carrega o CSV
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

# Mostra as primeiras linhas
print("\n📊 Resultados Carregados:")
print(df)

# Estatísticas básicas
mean_acc = df["accuracy"].mean()
std_acc = df["accuracy"].std()

print(f"\n✅ Média da Acurácia: {mean_acc:.4f}")
print(f"📉 Desvio Padrão: {std_acc:.4f}")

# Gráfico
plt.figure(figsize=(8, 5))
plt.plot(df["seed"], df["accuracy"], marker='o', linestyle='-', color='blue', label='Accuracy por seed')
plt.axhline(mean_acc, color='green', linestyle='--', label=f'Média: {mean_acc:.4f}')
plt.title(f"Acurácia por Seed - {df['dataset'][0]}")
plt.xlabel("Seed")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_plot.png")
plt.show()

print("\n📈 Gráfico salvo em: results/accuracy_plot.png")
