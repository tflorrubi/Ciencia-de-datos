import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")
print("Columnas:", df.columns.tolist())
print(df.head())
X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
Y = df["variety"]
model = DecisionTreeClassifier(random_state=42)
model.fit(X, Y)

# 5. Importancia de variables
importances = model.feature_importances_
variables = X.columns

df_importance = pd.DataFrame({
    "Variable": variables,
    "Importancia": importances
}).sort_values(by="Importancia", ascending=False)

print("\n=== IMPORTANCIA DE VARIABLES ===")
print(df_importance)

# 6. Gráfica
plt.figure(figsize=(8,5))
plt.barh(df_importance["Variable"], df_importance["Importancia"])
plt.xlabel("Importancia")
plt.ylabel("Variables")
plt.title("Feature Importance - Árbol de Decisión (Iris)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()