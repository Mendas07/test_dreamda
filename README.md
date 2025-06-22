# 🧠 DreamDA - ResNet50 Training Benchmark

Este repositório contém experimentos de benchmark com **ResNet50** usando o repositório original [DreamDA](https://github.com/yunxiangfu2001/DreamDA), com melhorias para:

- Treinamento com **modelo pré-treinado do ImageNet**
- Salvamento de **resultados por seed**
- Análise de **média, desvio padrão** e geração de gráficos
- Organização compatível com execução via **VSCode ou terminal Linux**

---

## 📁 Estrutura esperada dos dados

Os datasets devem ser organizados dentro da pasta `datasets/`, e **cada conjunto de dados deve conter as subpastas `train/` e `test/`**, conforme abaixo:

```
datasets/
├── pets/
│   ├── train/
│   │   ├── class1/
│   │   └── class2/
│   └── test/
│       ├── class1/
│       └── class2/
├── shenzhen/
├── caltech101/
├── cars/
└── stl10/
```

---

## 🚀 Como executar

### 1. Crie e ative o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute o treinamento

Edite o arquivo `train_resnet50.py` para selecionar o dataset desejado:

```python
DATASET_NAME = "shenzhen"  # ou "pets", "cars", etc.
DATASET_PATH = f"datasets/{DATASET_NAME}"
```

Execute:

```bash
python train_resnet50.py
```

Os resultados por seed serão salvos em:
```
results/shenzhen.csv
```

---

### 3. Execute a análise de resultados

```bash
python analyze_results.py
```

Esse script gera:
- Estatísticas no terminal (média, desvio padrão)
- Um gráfico salvo em `results/accuracy_plot.png`

---

## 📝 Scripts disponíveis

| Script | Função |
|--------|--------|
| `train_resnet50.py` | Treina a ResNet50 com pesos do ImageNet em um dataset escolhido |
| `save_results_only.py` | Salva manualmente os resultados de acurácia (modo offline) |
| `analyze_results.py` | Lê os `.csv` de resultados e gera estatísticas + gráfico |

---

## 📌 Requisitos

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- pandas

Instale tudo com:
```bash
pip install -r requirements.txt
```

Se não tiver o arquivo `requirements.txt`, crie com:
```txt
torch
torchvision
matplotlib
pandas
```

---

## 📦 Sobre o repositório original

Este projeto é baseado em:
🔗 https://github.com/yunxiangfu2001/DreamDA

A estrutura foi adaptada para facilitar testes locais e controle de experimentos com foco em modelos pré-treinados.

---

## 🤝 Contribuição

Sinta-se à vontade para forkar, abrir issues ou contribuir com melhorias.

---

## 🧑‍💻 Autor

Gabriel Anastacio — Engenharia de Automação ✨
