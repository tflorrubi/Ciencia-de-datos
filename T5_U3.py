import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset sin etiquetas
data = {
    "ingreso_anual": [20,25,30,35,40,45,100,110,120,130,55,65,70,80],
    "gasto_anual":   [15,18,20,22,23,25,70,75,80,85,35,40,45,50]
}

df = pd.DataFrame(data)

# Paso 1 – Seleccionar variables
X = df[["ingreso_anual", "gasto_anual"]]

# Método del codo (probar 3 valores de K o más)
inertia = []
Ks = [2,3,4,5,6]

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

# Graficar el método del codo
plt.plot(Ks, inertia, marker="o")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.show()

# Paso 2 – Aplicar K-Means con K óptimo (ej: 3)
k_optimo = 3
kmeans = KMeans(n_clusters=k_optimo, random_state=42)
clusters = kmeans.fit_predict(X)
df["cluster"] = clusters

# Paso 3 – Gráfica de clusters
plt.scatter(df["ingreso_anual"], df["gasto_anual"], c=df["cluster"])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, marker="x")
plt.xlabel("Ingreso anual")
plt.ylabel("Gasto anual")
plt.title("Clusters formados por K-Means")
plt.show()

print(df)