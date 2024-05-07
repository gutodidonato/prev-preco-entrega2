import pandas as pd
import numpy as np
import random

df = pd.read_excel("predicao.xlsx")
df["Preço Previsto"] = df["Preço Normal"] * df["Aprovação"]


print(df.info())
df["Preço Previsto"] = np.where(
    df["Preço Previsto"] < df["Custo Produção"],
    df["Custo Produção"] + 2,
    df["Preço Previsto"],
)


# utilizei a biblioteca random para gerar o preço normal/aprovação e custo produção

print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X = df[["Preço Normal", "Aprovação", "Custo Produção"]]
y = df["Preço Previsto"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
