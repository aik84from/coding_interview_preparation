# Coding interview preparation

Часто задаваемые вопросы: https://aik84from.github.io/faq.html

(C) Калинин Александр Игоревич


### [python] Как загрузить данные из файла в формате CSV?

```python
import pandas as pd


df = pd.read_csv("example.csv")
```


### [python] Как посмотреть общую информацию о данных?

```python
df.info()
```


### [python] Как отобразить первые 10 строк?

```python
df.head(10)
```


### [python] Как отобразить крайние 10 строк?

```python
df.tail(10)
```


### [python] Как отобразить случайные 10 строк?

```python
df.sample(10)
```


### [python] Как узнать описательную статистику?

```python
df.describe()
```


### [python] Как посмотреть медиану с учётом группы?

```python
df.groupby(by="target").median()
```


### [python] Как посмотреть линейную корреляцию?

```python
df[["alpha", "beta"]].corr()
```


### [python] Как посмотреть гистограмму распределения?

```python
import matplotlib.pyplot as plt


df["alpha"].hist(bins=50)
plt.show()
```


### [python] Как отобразить график?

```python
df["alpha"].plot()
plt.title("title")
plt.xlabel("xlabel") 
plt.ylabel("ylabel") 
plt.grid(True) 
plt.show()
```


### [python] Как визуально отобразить зависимость двух переменных?

```python
plt.scatter(df["alpha"], df["beta"])
plt.show()
```


### [python] Как выполнить сортировку по убыванию?

```python
df.sort_values(by=["alpha"], ascending=False).head(10)
```


### [python] Как применить формулу, например, найти кинетическую энергию? (1)

```python
df["kinetic_energy"] = (df["alpha"] * (df["beta"]**2)) / 2
```


### [python] Как применить формулу, например, найти кинетическую энергию? (2)

```python
import numpy as np


mass = np.linspace(1.0, 10.0, num=100)
velocity = np.linspace(1.0, 10.0, num=100)

# Показанный вариант более быстрый
kinetic_energy = 0.5 * (mass * np.power(velocity, 2))
```


### [python] Как сохранить таблицу в файл?

```python
result = df[["alpha", "beta", "kinetic_energy"]]

result.to_csv("result.csv", index=False)
result.to_json("result.json", orient="records", indent=4)
result.to_excel("result.xlsx")
```


### [python] Как получить данные из внешних API? (1)

```python
import requests


try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(data)
except requests.exceptions.ConnectionError as e:
    print(e)
except requests.exceptions.HTTPError as e:
    print(e)
except requests.exceptions.RequestException as e:
    print(e)

```


### [python] Как получить данные из внешних API? (2)

```python
import httpx


try:
    response = httpx.get(url)
    response.raise_for_status()
    data = response.json()
    print(data)
except httpx.RequestError as e:
    print(e)
except httpx.HTTPStatusError as e:
    print(e)
except Exception as e:
    print(e)

```


### [python] Как получить данные из внешних API? (3)

```python
# Возможно, единичные локальные запросы проще делать в bash
# wget -O example https://example.com
```


### [python] Как получить данные из внешних API? (4)

```python
# Возможно, единичные локальные запросы проще делать в bash
# curl -o example http://www.example.com
```


### [python] Как воспользоваться кластерным анализом?

```python
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=2).fit(df[["alpha", "beta"]])
print(kmeans.cluster_centers_)
print(kmeans.labels_)
```


### [python] Как воспользоваться классификатором?

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


X = df[["alpha", "beta"]]
target = df["target"]

model = RandomForestClassifier(n_estimators=1500, max_depth=200)
print(cross_val_score(model, X, target, cv=5, scoring="f1"))
```


### [python] Как применить формулу (на ваш выбор) с помощью Polars?

```python
import polars as pl


df = pl.read_csv("example.csv").select(
    (pl.col("alpha") - pl.col("beta")).alias("diff"),
    pl.col("target").alias("target")
)

ctx = pl.SQLContext(data=df, eager=True)
print(ctx.execute("SELECT * FROM data LIMIT 5;"))
```


### [python] Как написать функцию Root Mean Squared Error (RMSE)?

```python
import numpy as np
from sklearn.metrics import root_mean_squared_error


y_pred = np.array([8, 4, 9, 7, -9, 0, 2, 5, -8, -6, -4, -2, -1, 3])
y_true = np.array([0, -7, -5, -8, -1, -4, 1, 6, 5, 3, 8, -2, 7, -6])

print(root_mean_squared_error(y_true,  y_pred))
```


