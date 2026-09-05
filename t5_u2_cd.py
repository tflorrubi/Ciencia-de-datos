#TAREA 5
#FLOR RUBI TORRES DELGADO
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


df = pd.read_csv("iris.csv")
X_original = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
Y = df["variety"]
X_train_o, X_test_o, y_train, y_test = train_test_split(
    X_original, Y, test_size=0.3, random_state=42
)

# PCA para reducir a 2 componentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

X_train_r, X_test_r, _, _ = train_test_split(
    X_reduced, Y, test_size=0.3, random_state=42
)

# Modelo con dataset original
start_o = time.time()
model_original = DecisionTreeClassifier(random_state=42)
model_original.fit(X_train_o, y_train)
pred_o = model_original.predict(X_test_o)
end_o = time.time()

accuracy_original = accuracy_score(y_test, pred_o)
tiempo_original = end_o - start_o

# Modelo con dataset reducido
start_r = time.time()
model_reducido = DecisionTreeClassifier(random_state=42)
model_reducido.fit(X_train_r, y_train)
pred_r = model_reducido.predict(X_test_r)
end_r = time.time()

accuracy_reducido = accuracy_score(y_test, pred_r)
tiempo_reducido = end_r - start_r

# Resultados
print("\n=== RESULTADOS MODELO ORIGINAL ===")
print("Accuracy:", accuracy_original)
print("Tiempo:", tiempo_original)

print("\n=== RESULTADOS MODELO REDUCIDO ===")
print("Accuracy:", accuracy_reducido)
print("Tiempo:", tiempo_reducido)

print("\n=== TABLA COMPARATIVA ===")
print("Métrica\t\tOriginal\tReducido")
print(f"Accuracy\t{accuracy_original:.4f}\t\t{accuracy_reducido:.4f}")
print(f"Tiempo (s)\t{tiempo_original:.6f}\t{tiempo_reducido:.6f}")
