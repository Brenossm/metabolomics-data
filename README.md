# metabolomics-data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Carregue os dados em um DataFrame
df = pd.read_csv('/content/MATRIZ_POS_&_NE.csv')

print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

df.drop('samples', axis=1, inplace=True)

df.drop('class', axis=1, inplace=True)


scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df)

df = df.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['alvo'] = y


sns.scatterplot(x='PC1', y='PC2', hue='alvo', data=df_pca)
plt.show()
