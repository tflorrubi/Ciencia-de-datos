import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("iris.csv")
print("Columnas del dataset:")
print(df.columns.tolist())

X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=4)
pca.fit(X_scaled)
#Varianza
varianza = pca.explained_variance_ratio_
varianza_acumulada = varianza.cumsum()
print("\n=== VARIANZA EXPLICADA POR CADA COMPONENTE ===")
for i, v in enumerate(varianza):
    print(f"Componente {i+1}: {v:.4f}")

print("\n=== VARIANZA ACUMULADA ===")
for i, v in enumerate(varianza_acumulada):
    print(f"Hasta Componente {i+1}: {v:.4f}")

#Gráfica de varianza acumulada
plt.figure(figsize=(7,5))
plt.plot(range(1, 5), varianza_acumulada, marker='o')
plt.title("Varianza Acumulada - PCA (Iris)")
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Acumulada")
plt.grid(True)
plt.show()

#Mostrar componentes principales
componentes = pd.DataFrame(
    pca.components_,
    columns=["sepal.length", "sepal.width", "petal.length", "petal.width"],
    index=[f"PC{i+1}" for i in range(4)]
)

print("\n=== MATRIZ DE COMPONENTES PRINCIPALES ===")
print(componentes)