# Code Snippets

## Загрузка файла

Утилита wget:

```bash
wget -O dataset.csv http://127.0.0.1:1984/dataset.csv
```

Утилита curl:

```bash
curl -o dataset.csv http://127.0.0.1:1984/dataset.csv
```

Контрольная сумма md5:

```bash
md5sum dataset.csv
```

Контрольная сумма sha1:

```bash
sha1sum dataset.csv
```

## Работа с логом (журналом)

Просмотр первых пяти строк:

```bash
head -n 5 dataset.csv
```

Просмотр последних пяти строк:

```bash
tail -n 5 dataset.csv
```

Просмотр всего файла (страницами):

```bash
cat dataset.csv | less
```

Просмотр всего файла в обратном порядке:

```bash
tac dataset.csv
```

Поиск подстроки:

```bash
cat dataset.csv | grep "255"
```

## Запрос на API (JSON)

С помощью curl (форматирование и подсветка синтаксиса JSON)

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"prompt": "debug=True"}' http://127.0.0.1:8084/ | jq .
```

С помощью скрипта на Python:

```python
import requests

url = "http://127.0.0.1:8084/"
promts = [
    {"prompt": "debug=True"},
    {"prompt": "example"}
]

for promt in promts:
    response = requests.post(url, json=promt)
    print(response.json())
```

## Интерактивный анализ данных

Импорт необходимых модулей:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Загрузка данных:

```python
df = pd.read_csv("dataset.csv")
```

Общая информация о данных:

```python
df.info()
```

Первые пять наблюдений:

```python
df.head(5)
```

Последние пять наблюдений:

```python
df.tail(5)
```

Случайные пять наблюдений:

```python
df.sample(5)
```

Описательная статистика:

```python
df.describe()
```

Среднее с учётом группы:

```python
df.groupby("target").mean()
```

Отображение гистограммы:

```python
median = df["a"].median()
plt.hist(df["a"], bins=50)
plt.title(f"median = {median}")
plt.show()
```

Отображение графика:

```python
plt.plot(df["a"])
plt.xlabel("xlabel example")
plt.ylabel("ylabel example")
plt.title("title example")
plt.show()
```

## Кластерный анализ данных

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


plt.style.use("bmh")
colors = np.array(["red", "green"])

df = pd.read_csv("dataset.csv")
X = StandardScaler().fit_transform(df[["a", "b"]])

clusters = DBSCAN(eps=0.4, min_samples=50).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=colors[clusters])
plt.show()
```

Классификатор Random Forest:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_csv("dataset.csv")

X = df[["a", "b"]]
target = df["target"]

model = RandomForestClassifier(n_estimators=1500, max_depth=200)
print(cross_val_score(model, X, target, cv=5, scoring="f1"))
```

## Окружение для анализа данных

Создать виртуальное окружение Python:

```bash
python3 -m venv venv
```

Задействовать виртуальное окружение Python:

```bash
source venv/bin/activate
```

Установить модули для анализа данных:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost fastapi "uvicorn[standard]" Django djangorestframework django-filter markdown psycopg2-binary jinja2 requests pytest httpx
```

Как посмотреть настройки сетевой карты?

```bash
ip address show
```

Как запустить встроенный в Python сервер?

```bash
python3 -m http.server 1984
```

## Локальная установка PostgreSQL

Как установить Podman?

```bash
sudo apt install -y podman
```

Как загрузить образ PostgreSQL?

```bash
podman pull docker.io/postgres
```

Как подготовить и запустить контейнер? 

**Важно:** подразумевается локальная установка для целей разработки или тестирования. Не для ПРОМ!

```bash
podman run -dt --name devdb -e POSTGRES_PASSWORD=dev -e POSTGRES_USER=dev -e POSTGRES_DB=dev -p 5432:5432 postgres
```

Как запустить уже существующий контейнер?

```bash
podman start devdb
```

Как зайти в консольный клиент PostgreSQL?

```bash
podman exec -it devdb psql -U dev -d dev
```

Как остановить контейнер?

```bash
podman stop devdb
```

Как в Podman выполнить очистку?

```bash
podman system prune --all --force
```

Как узнать версию PostgreSQL?

```sql
SELECT VERSION();
```

Как посмотреть размер таблицы?

```sql
SELECT pg_size_pretty(pg_relation_size('example')) relation_size;
```

Как выполнить VACUUM ANALYZE?

```sql
VACUUM ANALYZE example;
```

Как узнать количество страниц и кортежей?

```sql
SELECT relpages, reltuples FROM pg_class WHERE relname = 'example';
```

## Простейший мониторинг ресурсов компьютера или сервера

Как узнать свободное место на диске?

```bash
df -h
```

Как узнать размер каталогов?

```bash
du -h
```

Как посмотреть свободную оперативную память?

```bash
free -h
```

Как посмотреть процессы?

```bash
ps -A
```

Как посмотреть нагрузку на процессор и память?

* Клавиша P — использование CPU (сортировка)
* Клавиша M — использование ОЗУ (сортировка)
* Клавиша h — справка
* Клавиша q — выход

```bash
top
```

Как завершить процесс (pid = 123) вежливо?

```bash
kill -15 123
```

Как завершить процесс (pid = 123) принудительно?

```bash
kill -9 123
```


