#Tarea 4
#TorresDelgado Flor Rubi_21180991
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
# Cargar dataset
data = pd.read_csv('iris.csv')
# Seleccionar solo columnas numéricas
data_num = data.select_dtypes(include=['int64', 'float64'])
# PCA
pca = PCA()
pca.fit(data_num)
# Número de componentes
n_componentes = len(pca.explained_variance_ratio_)
print(f"Numero de componentes principales: {n_componentes}")
# Varianza explicada por cada componente (CORRECCIÓN AQUÍ)
varianza_explicada = pca.explained_variance_ratio_ * 100
print("Varianza explicada por cada componente (%):")
print(varianza_explicada)
# Gráfico de varianza acumulada
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(varianza_explicada), marker='o')
plt.xlabel('Componentes')
plt.ylabel('Varianza Acumulada (%)')
plt.title('Varianza Acumulada por PCA')
plt.grid(True)
plt.show()
# Componentes principales
componentes = pca.components_
print("Componentes principales:")
print(componentes)
# Determinar número de componentes para al menos 80% de varianza
n_componentes_conservados = np.where(np.cumsum(varianza_explicada) >= 80)[0][0] + 1
print(f"Componentes conservados: {n_componentes_conservados}")
print(f"Varianza explicada acumulada: {np.sum(varianza_explicada[:n_componentes_conservados]):.2f}%")
# Porcentaje de reducción dimensional
reduccion = (1 - n_componentes_conservados / n_componentes) * 100
print(f"Reducción del dataset: {reduccion:.2f}%")



