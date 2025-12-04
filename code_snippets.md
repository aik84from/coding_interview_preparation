# Useful Code Snippets

## Основные команды для загрузки файла

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

## Несколько команд для работы с текстовым файлом

Просмотр первых пяти строк:

```bash
head -n 5 dataset.csv
```

Просмотр последних пяти строк:

```bash
tail -n 5 dataset.csv
```

Просмотр всего файла (отдельными страницами):

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

## Интерактивный анализ данных на Python

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

Отображение гистограммы (в заголовке показана медиана):

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

## Кластерный анализ данных (Python)

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

## Классификатор Random Forest (Python)

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

## Классификация с помощью CatBoost (Python)

```python
import pandas as pd
from catboost import CatBoostClassifier


# Набор данных для обучения (размеченные признаки)
df = pd.read_csv("main.csv")

# Создание и обучение модели
model = CatBoostClassifier(iterations=1000, depth=2, learning_rate=0.1)
model.fit(df[["a", "b"]], df["target"])

# Сохранение для последующей проверки
model.save_model('model_v0')
```

## Загрузка бинарного файла

Функция на Python с применением httpx:

```python
import httpx


def download_file(url, file_name):
    try:
        with httpx.stream("GET", url) as response:
            response.raise_for_status() 
            with open(file_name, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
    except httpx.RequestError as e:
        print(e)
    except httpx.HTTPStatusError as e:
        print(e)
    except IOError as e:
        print(e)
```

Функция на Python с применением requests:

```python
import urllib3
import requests


def download_file(url, file_name):
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Status code: {response.status_code}")
    except requests.RequestException as e:
        print(e)
    except Exception as e:
        print(e)
```

Функция на чистом Go:

```go
package main

import (
    "os"
    "io"
    "fmt"
    "net/http"
)

func DownloadFile(url, fileName string) error {
    resp, err := http.Get(url)
    if err != nil {
        return fmt.Errorf("ERROR #001: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != 200 {
        return fmt.Errorf("ERROR #002: %d", resp.StatusCode)
    }
    
    outFile, err := os.Create(fileName)
    if err != nil {
        return fmt.Errorf("ERROR #003: %v", err)
    }
    defer outFile.Close()

    _, err = io.Copy(outFile, resp.Body)
    if err != nil {
        return fmt.Errorf("ERROR #004: %v", err)
    }

    return nil
}
```

## Запрос на API (JSON)

С помощью curl (jq используется для форматирования и подсветки синтаксиса JSON)

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
pip install pytest polars numpy pandas matplotlib seaborn scikit-learn catboost httpx requests psycopg2-binary
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

Важно: подразумевается локальная (не для ПРОМ) установка для целей разработки или тестирования.

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

```bash
# Клавиша P — использование CPU (сортировка)
# Клавиша M — использование ОЗУ (сортировка)
# Клавиша h — справка
# Клавиша q — выход
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

## Простейший способ делиться данными в локальной сети

Как посмотреть настройки сетевой карты?

```bash
ip address show
```

Важно: используйте показанные далее примеры кода только для проверенной безопасной сети, а лучше выполняйте копирование по SCP.

Как запустить встроенный в Python сервер?

```bash
python3 -m http.server 1984
```

Как написать на Python простой API?

```python
from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def example():
    return {"message": "Спасибо, что читаете мой блог!"}
```

Как написать на Go простой API?

```go
package main

import (
    "log"
    "net/http"
    "encoding/json"
)

func handler(w http.ResponseWriter, r *http.Request) {
    response := map[string]string{"message": "Спасибо, что читаете мой блог!"}
    w.Header().Set("Content-Type", "application/json")
    if err := json.NewEncoder(w).Encode(response); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
    }
}

func main() {
    http.HandleFunc("/", handler)
    log.Fatal(http.ListenAndServe(":1984", nil))
}
```

