import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

PLOTS_DIR = 'pipeline/plots/eda'
os.makedirs(PLOTS_DIR, exist_ok=True) 

# Leitura dos dados
ocorr = pd.read_csv('pipeline/data/ocorrencias.csv', sep=',')
# mudei o nome do dataset para dados_clinma_inmet pra facilitar e utf-8 p padronização
clima = pd.read_csv('pipeline/data/dados_clima_INMET.csv', sep=';', encoding='utf-8')
# renomeação das colunas para facilitar merge dos datasets
clima = clima.rename(columns={
    'Data Medicao': 'data',
    'PRECIPITACAO TOTAL, DIARIO (AUT)(mm)': 'chuva_mm',
    'TEMPERATURA MEDIA, DIARIA (AUT)(°C)': 'temp_media',
    'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)': 'umidade'
})

# Merge dos datasets
df = pd.merge(ocorr, clima, 
    on='data', 
    how='left', 
    suffixes=('_ocorr', '_clima')
)

df = df.dropna(subset=['chuva_mm_clima', 'temp_media', 'umidade'])
df = df.rename(columns={
    'chuva_mm_clima': 'chuva_mm'
})


print(df.info())
print(df.describe(include='all'))


plt.figure(figsize=(10,6))
sns.countplot(data=df, x='tipo_ocorrencia', order=df['tipo_ocorrencia'].value_counts().index)
plt.title('Distribuição de Ocorrências por Tipo')
plt.xticks(rotation=45)
plt.tight_layout() 
plt.savefig(os.path.join(PLOTS_DIR, 'ocorrencias_por_tipo.png'))
plt.show()


df['data'] = pd.to_datetime(df['data'], errors='coerce')
df['ano_mes'] = df['data'].dt.to_period('M')
plt.figure(figsize=(12,6))
df.groupby('ano_mes').size().plot(kind='line', marker='o')
plt.title('Ocorrências ao Longo do Tempo (Por Mês - Dados Brutos)')
plt.ylabel('Quantidade de Ocorrências')
plt.xlabel('Mês/Ano')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ocorrencias_ao_longo_do_tempo_brutas.png'))
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='tipo_ocorrencia', y='chuva_mm')
plt.title('Distribuição de Chuva por Tipo de Ocorrência')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'chuva_por_tipo_ocorrencia.png'))
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df[['chuva_mm', 'temp_media', 'umidade']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlação entre Variáveis Climáticas')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'heatmap_correlacao_climaticas.png'))
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data=df, x='bairro', order=df['bairro'].value_counts().index)
plt.title('Ocorrências por Bairro')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ocorrencias_por_bairro.png'))
plt.show()


df_pivot = df.groupby(['ano_mes', 'tipo_ocorrencia']).size().unstack(fill_value=0)
plt.figure(figsize=(15,8)) 
df_pivot.plot(kind='line', marker='o', ax=plt.gca()) 
plt.title('Evolução Mensal das Ocorrências por Tipo')
plt.ylabel('Quantidade de Ocorrências')
plt.xlabel('Mês/Ano')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Tipo de Ocorrência', bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ocorrencias_por_tipo_ao_longo_do_tempo.png'))
plt.show()

# Salvar CSV (já existente)
df.to_csv('pipeline/data/df_merged.csv', index=False)


df['ocorrencia_critica'] = df['gravidade'].apply(lambda x: 1 if x == 'alta' else 0)
df['n_ocorrencias_daily'] = df.groupby('data')['tipo_ocorrencia'].transform('count')

df_daily_for_target = df.groupby('data').agg({
    'n_ocorrencias_daily': 'max' 
}).reset_index()

target_values = df_daily_for_target['n_ocorrencias_daily'].values

plt.figure(figsize=(10, 6))
sns.histplot(target_values, bins=range(int(np.max(target_values)) + 2), kde=False, color='skyblue', edgecolor='black')
plt.title('Distribuição do Número Diário de Ocorrências (Variável Alvo)')
plt.xlabel('Número de Ocorrências por Dia')
plt.ylabel('Frequência de Dias')
plt.xticks(range(0, int(np.max(target_values)) + 2, 1 if int(np.max(target_values)) < 10 else 2))
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'distribuicao_n_ocorrencias_target.png'))
plt.show()

print(f"\nEstatísticas da variável alvo (n_ocorrencias):")
print(f"  Média: {np.mean(target_values):.2f}")
print(f"  Mediana: {np.median(target_values):.2f}")
print(f"  Máximo: {np.max(target_values):.2f}")
print(f"  Número de dias com 0 ocorrências: {np.sum(target_values == 0)} ({np.sum(target_values == 0)/len(target_values)*100:.2f}%)")
print(f"  Número de dias com 1 ocorrência: {np.sum(target_values == 1)} ({np.sum(target_values == 1)/len(target_values)*100:.2f}%)")

print("\nTodos os gráficos EDA foram salvos na pasta 'pipeline/plots/eda'.")
print("Alan Turing deu bença!! 🙏🙏")