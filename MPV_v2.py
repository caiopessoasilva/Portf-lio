import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Carregar os dados
print("Carregando os Dados...")
excel_location = r"C:\Users\caiop\OneDrive\Caio\Cocamar\cepr057v_CEP_24_060_EMP01.xlsx"
dataset = pd.read_excel(excel_location)

# Pré-processamento
print("Pré-processamento dos Dados...")
dataset['DATA EMISSAO'] = pd.to_datetime(dataset['DATA EMISSAO'], errors='coerce')
dataset['Ano Emissao'] = dataset['DATA EMISSAO'].dt.year
dataset['Mes Emissao'] = dataset['DATA EMISSAO'].dt.month
dataset['Dia Emissao'] = dataset['DATA EMISSAO'].dt.day

# Remover colunas não numéricas
dataset = dataset[['UNID', 'TT', 'COD.TRA', 'TIP.ESTQ', 'ITEM', 'CULTURA', 'SALDO QTD', 'FABRICANTE', 'DATA EMISSAO', 'Ano Emissao', 'Mes Emissao', 'Dia Emissao']]

# Filtrando Produtores com Mínimo de 30 Compras e 15 Dias Diferentes
print("Filtrando Produtores com Mínimo de 30 Compras e 15 Dias Diferentes...")
produtor_counts = dataset.groupby(['TT', 'COD.TRA']).size()
produtor_dias = dataset.groupby(['TT', 'COD.TRA'])['DATA EMISSAO'].nunique()

produtores_filtrados = produtor_counts[(produtor_counts >= 30) & (produtor_dias >= 15)].index
dataset_filtrado = dataset.set_index(['TT', 'COD.TRA']).loc[produtores_filtrados].reset_index()

# Preparando os dados para previsão da semana de compra
print("Preparando os dados para previsão da semana de compra...")
dataset_filtrado['Semana_Prevista'] = pd.to_datetime(dataset_filtrado['DATA EMISSAO']).dt.isocalendar().week

columns_to_drop = ['DATA EMISSAO', 'Ano Emissao', 'Mes Emissao', 'Dia Emissao']

X_semana = dataset_filtrado.drop(columns=columns_to_drop)
y_semana = dataset_filtrado['Semana_Prevista']

# Treinando o modelo para previsão da semana de compra
print("Treinando o Modelo RandomForest para previsão da semana...")
X_train_semana, X_test_semana, y_train_semana, y_test_semana = train_test_split(X_semana, y_semana, test_size=0.3, random_state=42)
rf_semana = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_semana.fit(X_train_semana, y_train_semana)

# Verificando a precisão do modelo
print("Verificando a precisão do modelo de previsão da semana...")
y_pred_semana = rf_semana.predict(X_test_semana)
semana_accuracy = r2_score(y_test_semana, y_pred_semana)
print(f"Acurácia (RandomForest Semana): {semana_accuracy}")

# Filtrando previsões para semanas de julho e agosto de 2024
print("Filtrando previsões para semanas de julho e agosto de 2024...")
df_julho_agosto = dataset_filtrado[(dataset_filtrado['Mes Emissao'] == 7) | (dataset_filtrado['Mes Emissao'] == 8)]
df_julho_agosto['Semana_Prevista'] = rf_semana.predict(df_julho_agosto.drop(columns=columns_to_drop))

if df_julho_agosto.empty:
    print("Nenhuma previsão foi gerada para julho e agosto de 2024.")
else:
    print(f"Total de previsões para julho e agosto: {len(df_julho_agosto)}")

# Preparando dados para previsão de produtos e quantidades
print("Preparando dados para previsão de produtos e quantidades...")
X_produto = df_julho_agosto.drop(columns=['ITEM', 'UNID', 'DATA EMISSAO', 'Ano Emissao', 'Mes Emissao', 'Dia Emissao'])
y_produto = df_julho_agosto['ITEM']

# Treinando o modelo para previsão de produtos e quantidades
print("Treinando o Modelo RandomForest para previsão de produtos e quantidades...")
rf_produto = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_produto.fit(X_produto, y_produto)
df_julho_agosto['Saldo_QTD_Pred'] = rf_produto.predict(X_produto)

# Gerando o arquivo CSV com as previsões
print("Criando arquivo CSV para as previsões...")
resultado = df_julho_agosto[['TT', 'COD.TRA', 'UNID', 'ITEM', 'Saldo_QTD_Pred', 'Semana_Prevista']]
output_path = r"C:\Users\caiop\OneDrive\Caio\Cocamar\Predicao_Julho_Agosto_2024_RandomForest.csv"
resultado.to_csv(output_path, index=False, encoding='utf-8')

print(f"Previsões salvas em {output_path} com sucesso!")
