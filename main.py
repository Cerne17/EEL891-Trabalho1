import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

df_treinamento = pd.read_csv('./conjunto_de_treinamento.csv')
df_teste = pd.read_csv('./conjunto_de_teste.csv')

df_treinamento = df_treinamento.drop(columns=[
    'id_solicitante',
    'possui_telefone_residencial',
    'possui_telefone_celular',
    'possui_carro',
    'grau_instrucao',
    'meses_no_trabalho',
    'estado_onde_nasceu',
    'estado_onde_reside',
    'codigo_area_telefone_residencial',
    'codigo_area_telefone_trabalho',
    'estado_onde_trabalha'
])

df_teste = df_teste.drop(columns=[
    'id_solicitante',
    'possui_telefone_residencial',
    'possui_telefone_celular',
    'possui_carro',
    'grau_instrucao',
    'meses_no_trabalho',
    'estado_onde_nasceu',
    'estado_onde_reside',
    'codigo_area_telefone_residencial',
    'codigo_area_telefone_trabalho',
    'estado_onde_trabalha'
])

df_treinamento = df_treinamento.replace(" ", np.nan)
df_teste = df_teste.replace(" ", np.nan)

nulos_treinamento = df_treinamento.isna().sum()
nulos_treinamento = nulos_treinamento[nulos_treinamento > 0]
num_linhas = df_treinamento.shape[0]
percentual_nulos = nulos_treinamento.apply(lambda x:(x/num_linhas)*100)
decresc_percentual_nulos = pd.Series(percentual_nulos).sort_values(ascending=False)
for i in decresc_percentual_nulos:
    if i>=50:
        df_treinamento = df_treinamento.drop(columns=[decresc_percentual_nulos[decresc_percentual_nulos == i].index[0]])
df_treinamento[['sexo']] = df_treinamento[['sexo']].fillna(value='N')
for i in decresc_percentual_nulos:
    att = decresc_percentual_nulos[decresc_percentual_nulos == i].index[0]
    if i < 50 and att!='sexo':
        df_treinamento[[att]] = df_treinamento[[att]].fillna(value=df_treinamento[att].mean().round())

nulos_teste = df_teste.isna().sum()
nulos_teste = nulos_teste[nulos_teste > 0]
num_linhas = df_teste.shape[0]
percentual_nulos = nulos_teste.apply(lambda x:(x/num_linhas)*100)
decresc_percentual_nulos = pd.Series(percentual_nulos).sort_values(ascending=False)
for i in decresc_percentual_nulos:
    if i>=50:
        df_teste = df_teste.drop(columns=[decresc_percentual_nulos[decresc_percentual_nulos == i].index[0]])
df_teste[['sexo']] = df_teste[['sexo']].fillna(value='N')
for i in decresc_percentual_nulos:
    att = decresc_percentual_nulos[decresc_percentual_nulos == i].index[0]
    if i < 50 and att!='sexo':
        df_teste[[att]] = df_teste[[att]].fillna(value=df_teste[att].mean().round())

df_treinamento = pd.get_dummies(df_treinamento, columns=['sexo', 'forma_envio_solicitacao'])

df_teste = pd.get_dummies(df_teste, columns=['sexo', 'forma_envio_solicitacao'])

binarizador = LabelBinarizer()

for col in ['possui_telefone_trabalho', 'vinculo_formal_com_empresa']:
    df_treinamento[col] = binarizador.fit_transform(df_treinamento[col])
    df_teste[col] = binarizador.fit_transform(df_teste[col])

df_treinamento_final = df_treinamento[df_treinamento.renda_extra != df_treinamento.renda_extra.max()]

y_treinamento = df_treinamento_final['inadimplente']
X_treinamento = df_treinamento_final.drop(columns=['inadimplente'])

modelo = RandomForestClassifier(
    criterion='gini',
    max_depth=15,
    max_features=10,
    min_samples_leaf=5,
    min_samples_split=0.001,
    n_estimators=200,
    random_state=0
)

modelo.fit(X_treinamento, y_treinamento)

resultados = modelo.predict(df_teste)
df_teste_original = pd.read_csv('./conjunto_de_teste.csv')
dict_resultados = {'inadimplente': resultados, 'id_solicitante': df_teste_original['id_solicitante']}
df_resultados = pd.DataFrame(dict_resultados)
df_resultados.to_csv('resultados.csv', index=False)