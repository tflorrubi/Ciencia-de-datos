import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset sencillo
data = {
    "horas_estudio": [1,2,3,4,5,6,7,8,9,10],
    "calificacion": ["R","R","R","R","A","A","A","A","A","A"]
}
df = pd.DataFrame(data)

# Paso 1 – Cargar dataset
# ya creado arriba

# Paso 2 – Variables
X = df[["horas_estudio"]]
y = df["calificacion"]

# Paso 3 – Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 4 – Modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Paso 5 – Predicciones
pred = knn.predict(X_test)

print("Predicciones:", pred)
print("Real:", list(y_test))
print("Accuracy:", accuracy_score(y_test, pred))

# Gráfica puntos
plt.scatter(X, [0 if c=="R" else 1 for c in y])
plt.xlabel("Horas de estudio")
plt.ylabel("Clase (0=R, 1=A)")
plt.title("Dataset clasificación")
plt.show()