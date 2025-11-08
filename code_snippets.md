# Code Snippets: SQL, Python, Go, Bash

## 1. Загрузка и просмотр набора данных

### [bash] Загрузка набора данных с помощью curl

```bash
curl -o dataset.csv http://127.0.0.1:1984/dataset.csv
```

### [bash] Загрузка набора данных с помощью wget

```bash
wget -O dataset.csv http://127.0.0.1:1984/dataset.csv
```

### [bash] Как определить контрольную сумму MD5?

```bash
md5sum dataset.csv
```

### [bash] Как определить контрольную сумму SHA1?

```bash
sha1sum dataset.csv
```

### [bash] Как посмотреть только первые 5 строк?

```bash
head -n 5 dataset.csv
```

### [bash] Как посмотреть только последние 5 строк?

```bash
tail -n 5 dataset.csv
```

### [bash] Как посмотреть содержимое файла?

```bash
cat dataset.csv | less
```

### [bash] Как посмотреть содержимое файла в обратном порядке?

```bash
tac dataset.csv
```

### [bash] Как получить уникальный список строк в файле?

```bash
cat dataset.csv | sort | uniq
```

### [bash] Как посчитать количество строк, слов и байт?

```bash
wc dataset.csv
```

## 2. Система управления базами данных

### [bash] Как установить Podman?

```bash
apt install -y podman
```

### [bash] Как загрузить образ PostgreSQL?

```bash
podman pull docker.io/postgres
```

### [bash] Как подготовить и запустить контейнер?

Важно: подразумевается локальная установка для целей разработки или тестирования.

```bash
podman run -dt --name devdb -e POSTGRES_PASSWORD=dev -e POSTGRES_USER=dev -e POSTGRES_DB=dev -p 5432:5432 postgres
```

### [bash] Как запустить уже существующий контейнер?

```bash
podman start devdb
```

### [bash] Как зайти в консольный клиент PostgreSQL?

```bash
podman exec -it devdb psql -U dev -d dev
```

### [bash] Как остановить контейнер?

```bash
podman stop devdb
```

### [bash] Как в Podman выполнить очистку?

```bash
podman system prune --all --force
```

### [sql] Как узнать версию PostgreSQL?

```sql
SELECT VERSION();
```

### [sql] Как посмотреть размер таблицы?

```sql
SELECT pg_size_pretty(pg_relation_size('example')) relation_size;
```

### [sql] Как выполнить VACUUM ANALYZE?

```sql
VACUUM ANALYZE example;
```

### [sql] Как узнать количество страниц и кортежей?

```sql
SELECT relpages, reltuples FROM pg_class WHERE relname = 'example';
```

## 3. Предварительный анализ данных

### [python] Посмотреть гистограмму для каждого столбца отдельно

```python
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("example.csv")

for column in df.columns:
    median = df[column].median()
    plt.hist(df[column], bins=50)
    plt.title(f"median({column}) = {median}")
    plt.show()
```

### [python] Предварительный кластерный анализ с помощью DBSCAN

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


plt.style.use("bmh")
colors = np.array(["red", "green"])

df = pd.read_csv("example.csv")
X = StandardScaler().fit_transform(df[["sensor", "beta"]])

clusters = DBSCAN(eps=0.4, min_samples=50).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=colors[clusters])
plt.show()
```

### [python] Классификатор Random Forest

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_csv("example.csv")

X = df[["sensor", "beta"]]
target = df["target"]

model = RandomForestClassifier(n_estimators=1500, max_depth=200)
print(cross_val_score(model, X, target, cv=5, scoring="f1"))
```

## 4. Минимальный мониторинг системных ресурсов

### [bash] Как узнать свободное место на диске?

```bash
df -h
```

### [bash] Как узнать размер каталогов?

```bash
du -h
```

### [bash] Как посмотреть память? (свободное место)

```bash
free -h
```

### [bash] Как посмотреть процессы?

```bash
ps -A
```

### [bash] Как посмотреть нагрузку на процессор и память?

```bash
# Клавиша P — использование CPU (сортировка)
# Клавиша M — использование ОЗУ (сортировка)
# Клавиша h — справка
# Клавиша q — выход
top
```

### [bash] Как завершить процесс вежливо?

```bash
kill -15 123
```

### [bash] Как завершить процесс принудительно?

```bash
kill -9 123
```

## 5. Окружение для анализа данных

### [bash] Создать виртуальное окружение Python

```bash
python3 -m venv venv
```

### [bash] Задействовать виртуальное окружение Python

```bash
source venv/bin/activate
```

### [bash] Установить модули для анализа данных

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost fastapi "uvicorn[standard]" Django djangorestframework django-filter markdown psycopg2-binary jinja2 requests
```

### [bash] Как посмотреть настройки сетевой карты?

```bash
ip address show
```

### [bash] Как запустить встроенный в Python сервер?

```bash
python3 -m http.server 8084
```

### [python] Простейший микросервис FastAPI

```python
from sensor import read_photoresistor
from fastapi import FastAPI, Depends


# Запуск: uvicorn api:app --reload
app = FastAPI()

@app.get("/photoresistor")
async def photoresistor(analog_value: float = Depends(read_photoresistor)):
    return {"analogValue": analog_value}

```

### [go] Простейший микросервис на Go

```go
package main

import (
    "net/http"
    "html/template"
)

type Content struct {
    Version string
}

// В документе example.html добавлен <h1>Version: {{.Version}}</h1>
var templates = template.Must(template.ParseFiles("example.html"))

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        content := Content{Version: "0.0.001"}
        templates.Execute(w, content)
    })
    http.ListenAndServe(":8084", nil)
}
```


