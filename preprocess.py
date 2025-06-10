# preprocess.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os 
# const da janela de tempo 
WINDOW_SIZE = 7
os.makedirs('pipeline/models', exist_ok=True)


df = pd.read_csv("pipeline/data/df_merged.csv", encoding="utf-8")
df['data'] = pd.to_datetime(df['data'], errors='coerce')
df = df.sort_values('data').reset_index(drop=True)

# agrupando o df para target e discorrer sobre features
df['n_ocorrencias'] = df.groupby('data')['tipo_ocorrencia'].transform('count')

df_daily = df.groupby('data').agg(
    # Chuva
    chuva_mm_mean=('chuva_mm', 'mean'),
    chuva_mm_sum=('chuva_mm', 'sum'),
    chuva_mm_max=('chuva_mm', 'max'),
    chuva_mm_min=('chuva_mm', 'min'),
    chuva_mm_std=('chuva_mm', 'std'),

    # Temperatura
    temp_media_mean=('temp_media', 'mean'),
    temp_media_max=('temp_media', 'max'),
    temp_media_min=('temp_media', 'min'),
    temp_media_std=('temp_media', 'std'),

    # Umidade
    umidade_mean=('umidade', 'mean'),
    umidade_max=('umidade', 'max'),
    umidade_min=('umidade', 'min'),
    umidade_std=('umidade', 'std'),
    
    # Target
    n_ocorrencias=('n_ocorrencias', 'max')
).reset_index()

# Transformar a variável alvo para log1p
df_daily['n_ocorrencias_transformed'] = np.log1p(df_daily['n_ocorrencias'])

# Criar features de lag
df_daily['n_ocorrencias_lag1'] = df_daily['n_ocorrencias_transformed'].shift(1)
df_daily['n_ocorrencias_lag7'] = df_daily['n_ocorrencias_transformed'].shift(7)

# Remover linhas com NaN
df_daily = df_daily.fillna(0) 

# Gerar janelas para o LSTM
features = [
    'chuva_mm_mean', 'chuva_mm_sum', 'chuva_mm_max', 'chuva_mm_min', 'chuva_mm_std',
    'temp_media_mean', 'temp_media_max', 'temp_media_min', 'temp_media_std',
    'umidade_mean', 'umidade_max', 'umidade_min', 'umidade_std',
    'n_ocorrencias_lag1', 'n_ocorrencias_lag7' 
]


def windows_lstm(df_, window_size=WINDOW_SIZE):
    X_list, y_list = [], []
    for i in range(len(df_) - window_size):
        X_list.append(df_[features].iloc[i: i+window_size].values)
        y_list.append(df_['n_ocorrencias_transformed'].iloc[i+window_size])
    return np.array(X_list), np.array(y_list)

X_lstm, y_lstm = windows_lstm(df_daily)

print(f'X_lstm shape: {X_lstm.shape} | y_lstm shape: {y_lstm.shape}')

# Split 60/20/20 (sem estratificação, pq é regressão)
n_samples = X_lstm.shape[0]
idx_split1 = int(n_samples * 0.6)
idx_split2 = int(n_samples * 0.8)

X_train = X_lstm[:idx_split1]
y_train = y_lstm[:idx_split1]

X_val = X_lstm[idx_split1:idx_split2]
y_val = y_lstm[idx_split1:idx_split2]

X_test = X_lstm[idx_split2:]
y_test = y_lstm[idx_split2:]

print('LSTM splits:')
print(f'   Train: {X_train.shape}, {y_train.shape}')
print(f'   Val:   {X_val.shape}, {y_val.shape}')
print(f'   Test:  {X_test.shape}, {y_test.shape}')

# Scaler
# O input_size agora será o número de features final
scaler = StandardScaler()
n_features_for_scaler = X_train.shape[2] # Pega o número de features final

X_train_flat = X_train.reshape(-1, n_features_for_scaler)
scaler.fit(X_train_flat)

def scaler_all(X, scaler):
    flat = X.reshape(-1, X.shape[2])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(X.shape)

X_train_scaled = scaler_all(X_train, scaler)
X_val_scaled = scaler_all(X_val, scaler)
X_test_scaled = scaler_all(X_test, scaler)

# Salvar datasets
np.save('pipeline/data/X_train_lstm.npy', X_train_scaled)
np.save('pipeline/data/y_train_lstm.npy', y_train) 
np.save('pipeline/data/X_val_lstm.npy', X_val_scaled)
np.save('pipeline/data/y_val_lstm.npy', y_val)
np.save('pipeline/data/X_test_lstm.npy', X_test_scaled)
np.save('pipeline/data/y_test_lstm.npy', y_test)

# Salvar scaler para usar em prod/deploy
joblib.dump(scaler, 'pipeline/models/scaler.pkl') # Garante que está na pasta 'models'

print('Datasets e scaler salvos em pipeline/data e pipeline/models, respectivamente.')
print("Alan Turing deu bença!! 🙏🙏")