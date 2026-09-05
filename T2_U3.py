import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "metros": [50, 60, 70, 80, 90, 100, 120, 150],
    "precio": [65000, 75000, 82000, 90000, 98000, 105000, 130000, 160000]
}
df = pd.DataFrame(data)
X = df[['metros']]
y = df['precio']

modelo = LinearRegression()
modelo.fit(X, y)

df['prediccion'] = modelo.predict(X)

# Gráfica
plt.scatter(df['metros'], df['precio'])
plt.plot(df['metros'], df['prediccion'])
plt.xlabel("Metros cuadrados")
plt.ylabel("Precio")
plt.title("Regresión Lineal")
plt.show()

print("Coeficiente (pendiente):", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)