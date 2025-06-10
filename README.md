# Previsão de Ocorrências Extremas com LSTM + Attention

Este projeto usa Deep Learning para prever ocorrências críticas relacionadas a clima extremo em São Paulo, com base em dados meteorológicos e históricos de incidentes.

##  Problema

Eventos climáticos extremos têm gerado ocorrências graves em áreas urbanas — deslizamentos, alagamentos e quedas de árvores. Antecipar esses eventos permite uma resposta mais rápida e eficiente por parte das autoridades.

##  Solução

Utilizei uma arquitetura de LSTM com mecanismo de atenção para prever a quantidade de ocorrências futuras, considerando séries temporais meteorológicas (chuva, temperatura, umidade, etc) e registros históricos.

## 📊 Dados

- 📍 **Ocorrências reais** por tipo e bairro (2022 a 2024)
- 🌦️ **Dados climáticos** (INMET)
- 🔗 **Merge temporal** por data/hora

##  Pipeline

- EDA completa com visualizações (`plots/eda/`)
- Pré-processamento com normalização e divisão temporal
- Modelo LSTM com atenção treinado e salvo (`models/lstm_attention.pt`)
- Visualizações de métricas e curvas de loss (`plots/lstm/`)
- Treinamento monitorado com TensorBoard (`runs/`)
- Scripts organizados em `eda.py`, `preprocess.py` e `train_lstm.py`

##  Resultados

- Modelo com boa capacidade de generalização temporal
- Detecção de padrões climáticos críticos correlacionados com incidentes
- Estrutura pronta para deployment em sistemas de monitoramento urbano

## 🧪 Tecnologias

- Python
- PyTorch
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- TensorBoard

## 📁 Estrutura

```

.
├── data/              # CSVs e datasets em .npy
├── models/            # Modelo treinado (.pt) e scaler
├── plots/             # Visualizações da EDA e LSTM
├── runs/              # Logs do TensorBoard
├── eda.py             # Análise exploratória
├── preprocess.py      # Pré-processamento
├── train_lstm.py      # Treinamento do modelo
└── requirements-pipeline.txt

```

## 👨‍💻 Autor
[@vinicius](https://www.linkedin.com/in/viniruggeri)  
Desenvolvedor Backend & Engenheiro de IA  
FIAP | 2025
