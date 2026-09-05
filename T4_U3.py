import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso

# Dataset con varias variables
data = {
    "publicidad_tv": [100,120,130,140,150,160,30,20,60,90],
    "publicidad_radio": [20,25,30,32,40,45,5,10,12,15],
    "publicidad_redes": [300,320,400,420,500,520,200,180,250,260],
    "ventas": [15,18,20,22,25,28,5,6,10,14]
}

df = pd.DataFrame(data)

X = df.drop("ventas", axis=1)
y = df["ventas"]

# Lineal
model_lin = LinearRegression()
model_lin.fit(X, y)

# Lasso
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X, y)

print("Coeficientes Regresión Lineal:", model_lin.coef_)
print("Coeficientes Lasso:", model_lasso.coef_)

# Gráfica comparativa
variables = X.columns
plt.bar(variables, model_lin.coef_, label="Lineal")
plt.bar(variables, model_lasso.coef_, alpha=0.7, label="Lasso")
plt.title("Comparación de coeficientes")
plt.legend()
plt.show()