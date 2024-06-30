# SQL, Python, Go

## Coding interview preparation

(C) 2024 Калинин Александр Игоревич

## SQL

### Как узнать версию PostgreSQL?

```sql
SELECT VERSION();
```

### Как создать таблицу?

```sql
CREATE TABLE example (
    id SERIAL PRIMARY KEY,
    visitors INT
);
```

### Как добавить данные в таблицу?

```sql
INSERT INTO example (id, visitors) VALUES
(1, 709),
(2, 749),
(3, 180),
(4, 518),
(5, 964),
(6, 180),
(7, 997),
(8, 562);
```

### Как обновить запись?

```sql
UPDATE example SET visitors = 100 WHERE id = 8;
```

### Как удалить запись?

```sql
DELETE FROM example WHERE id = 8;
```

### Как узнать количество?

```sql
SELECT COUNT(*) FROM example;
```

### Как найти среднее значение?

```sql
SELECT AVG(visitors) FROM example;
```

### Как найти максимальное и минимальное значение?

```sql
SELECT MAX(visitors), MIN(visitors) FROM example;
```

### Как найти дисперсию?

```sql
SELECT VARIANCE(visitors) FROM example;
```

### Как найти сумму?

```sql
SELECT SUM(visitors) FROM example;
```

### Как найти ТОП-5 самых больших значений?

```sql
SELECT * FROM example ORDER BY visitors DESC LIMIT 5;
```

### Как найти все значения, которые находятся в нужном интервале?

```sql
SELECT * FROM example WHERE visitors BETWEEN 190 AND 800;
```

### Как найти дубликаты?

```sql
SELECT
    COUNT(visitors),
    visitors
FROM example
GROUP BY visitors
HAVING COUNT(visitors) > 1;
```

### Как узнать ранг?

```sql
SELECT
    id,
    visitors,
    dense_rank() OVER (ORDER BY visitors DESC) AS rank
FROM example;
```

### Как написать обобщённое табличное выражение?

```sql
WITH data AS (
    SELECT ARRAY[1, 2, 3, 4] AS alpha
)

SELECT * FROM data;
```

### Как сделать сочетание двух строк?

```sql
SELECT 26 AS age
UNION ALL
SELECT 40 AS age;
```

### Как написать свою функцию?

```sql
CREATE FUNCTION example_f(n integer)
RETURNS integer AS $$
BEGIN
    RETURN factorial(n);
END; $$
LANGUAGE plpgsql;
```

### Как узнать размер таблицы?

```sql
SELECT pg_size_pretty(pg_relation_size('example')) relation_size;
```

### Как создать индекс?

```sql
CREATE INDEX idx_example_visitors ON example(visitors);
```

### Как получить план выполнения запроса?

```sql
EXPLAIN (ANALYZE)
SELECT * FROM example WHERE id IN (1, 5, 7);
```

### Как скопировать таблицу?

```sql
CREATE TABLE backup_example AS TABLE example;
```

### Как использовать полнотекстовый поиск?

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(256)
);

INSERT INTO documents (id, title) VALUES
(1, 'Better late than never'),
(2, 'A good man is hard to find'),
(3, 'Worrying never did anyone any good');

ALTER TABLE documents 
ADD COLUMN title_gin tsvector
GENERATED ALWAYS AS (to_tsvector('english', title)) STORED;

CREATE INDEX idx_title_gin ON documents USING GIN (title_gin);

SELECT *
FROM documents
WHERE title_gin @@ to_tsquery('english', 'never');
```

### Как найти данные с помощью регулярного выражения?

```sql
SELECT title, REGEXP_MATCHES(title, 'never') result FROM documents;
```

### Как найти длину строки и контрольную сумму?

```sql
SELECT title, LENGTH(title), MD5(title) FROM documents;
```

## Python

### Как загрузить данные из файла в формате CSV?

```python
import pandas as pd


df = pd.read_csv("example.csv")
```

### Как посмотреть общую информацию о данных?

```python
df.info()
```

### Как отобразить первые 10 строк?

```python
df.head(10)
```

### Как отобразить крайние 10 строк?

```python
df.tail(10)
```

### Как отобразить случайные 10 строк?

```python
df.sample(10)
```

### Как узнать описательную статистику?

```python
df.describe()
```

### Как посмотреть медиану с учётом группы?

```python
df.groupby(by="target").median()
```

### Как посмотреть линейную корреляцию?

```python
df[["alpha", "beta"]].corr()
```

### Как посмотреть гистограмму распределения?

```python
import matplotlib.pyplot as plt


df["alpha"].hist(bins=50)
plt.show()
```

### Как отобразить график?

```python
df["alpha"].plot()
plt.title("title")
plt.xlabel("xlabel") 
plt.ylabel("ylabel") 
plt.grid(True) 
plt.show()
```

### Как визуально отобразить зависимость двух переменных?

```python
plt.scatter(df["alpha"], df["beta"])
plt.show()
```

### Как выполнить сортировку по убыванию?

```python
df.sort_values(by=["alpha"], ascending=False).head(10)
```

### Как применить формулу, например, найти кинетическую энергию? (1)

```python
df["kinetic_energy"] = (df["alpha"] * (df["beta"]**2)) / 2
```

### Как применить формулу, например, найти кинетическую энергию? (2)

```python
import numpy as np


mass = np.linspace(1.0, 10.0, num=100)
velocity = np.linspace(1.0, 10.0, num=100)

# Показанный вариант более быстрый.
kinetic_energy = 0.5 * (mass * np.power(velocity, 2))
```

### Как сохранить таблицу в файл?

```python
result = df[["alpha", "beta", "kinetic_energy"]]

result.to_csv("result.csv", index=False)
result.to_json("result.json", orient="records", indent=4)
result.to_excel("result.xlsx")
```

### Как получить данные из внешних API? (1)

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

### Как получить данные из внешних API? (2)

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

### Как получить данные из внешних API? (3)

```python
# Возможно, единичные локальные запросы проще делать в bash
# wget -O example https://example.com
```

### Как получить данные из внешних API? (4)

```python
# Возможно, единичные локальные запросы проще делать в bash
# curl -o example http://www.example.com
```

### Как воспользоваться кластерным анализом?

```python
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=2).fit(df[["alpha", "beta"]])
print(kmeans.cluster_centers_)
print(kmeans.labels_)
```

### Как воспользоваться классификатором?

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


X = df[["alpha", "beta"]]
target = df["target"]

model = RandomForestClassifier(n_estimators=1500, max_depth=200)
print(cross_val_score(model, X, target, cv=5, scoring="f1"))
```

### Как применить формулу (на ваш выбор) с помощью Polars?

```python
import polars as pl


df = pl.read_csv("example.csv").select(
    (pl.col("alpha") - pl.col("beta")).alias("diff"),
    pl.col("target").alias("target")
)

ctx = pl.SQLContext(data=df, eager=True)
print(ctx.execute("SELECT * FROM data LIMIT 5;"))
```

## Golang

### Как найти расстояние между двумя точками на карте?

```go
package main

import (
    "fmt"
    "math"
)

type Point struct {
    x float64
    y float64
}

func (p Point) getEuclideanDistance(q Point) float64 {
    return math.Sqrt(math.Pow(q.x - p.x, 2) + math.Pow(q.y - p.y, 2))
}

func main() {
    result := Point{17, 22}.getEuclideanDistance(Point{36, 72})
    fmt.Println((result - 53.488) < 0.001)
}
```

### Как написать поиск элемента в HashMap со сложностью O(1)?

```go
package main

import "fmt"

type Url struct {
    id uint64
    score uint64
    url string
    title string
}

func main() {
    urls := make(map[string]Url)
    urls["august"] = Url{1, 5, "http://url_1", "August"}
    urls["september"] = Url{2, 8, "http://url_2", "September"}
    urls["october"] = Url{3, 9, "http://url_3", "October"}

    if item, ok := urls["september"]; ok {
        fmt.Println(item)
    } else {
        fmt.Println("Not found")
    }
}
```

### Как написать поиск элемента в Slice со сложностью O(n)?

```go
func linearSearch(items []int, target int) int {
    for i, val := range items {
        if val == target {
            return i
        }
    }
    return -1
}
```

### Как написать поиск элемента в Slice со сложностью O(log n)?

```go
func binarySearch(items []int, target int) int {
    // В отличии от линейного поиска items должен быть заранее отсортирован
    low, high := 0, len(items) - 1
    for low <= high {
        mid := low + (high-low)/2
        if items[mid] == target {
            return mid
        } else if items[mid] < target {
            low = mid + 1
        } else {
            high = mid - 1
        }
    }
    return -1
}
```

### Как эффективно искать запись в гигантских наборах данных?

```go
/*
Лучше всего сохранить нужные данные в отдельную таблицу.
Это можно сделать следующими способами:

+ На этапе обработки больших данных в Apache Spark.
+ Отслеживание очередей, например, Apache Kafka [1].
+ Фоновыми запросами в PostgreSQL, ClickHouse, Elasticsearch, Tarantool [2].
+ Забрать из кэша (Apache Ignite, Redis, Memcached) [3].

[1] — при гигантских потоках данных, которые сложно сразу направить в СУБД.
[2] — из нужного ДЦ (балансировка на уровне DNS).
[3] — если у вас есть кэширование, например, рейтинга лучших товаров.

Если нужен именно алгоритм поиска, то тут много вариантов.
Допустим, в качестве индекса создать матрицу вершин графа и обойти её.
Или выполнить простой линейный поиск по интервалам в матрице:
*/

package main

import (
    "fmt"
)

var matrix [4][2]int = [4][2]int{
    {7, 10},
    {4, 15},
    {1, 23},
    {20, 8},
}

func main() {
    x := 11
    for i := 0; i < 4; i++ {
        if matrix[i][0] <= x && matrix[i][1] >= x {
            fmt.Println("result", matrix[i])
        }
    }
}
```


