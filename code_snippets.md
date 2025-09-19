

# Code Snippets: SQL, Python, Go, Bash

## Free and open source

(C) Калинин Александр Игоревич


### [bash] Как посмотреть содержимое файла?

```bash
cat example_1.html | less
```

### [bash] Как посмотреть содержимое файла в обратном порядке?

```bash
tac example_1.html
```

### [bash] Как увидеть первые 5 строк?

```bash
head -n 5 example_1.html
```

### [bash] Как увидеть крайние 5 строк?

```bash
tail -n 5 example_1.html
```

### [bash] Как получить уникальный список строк в файле?

```bash
cat data.txt | sort | uniq
```

### [bash] Как посчитать количество строк, слов и байт?

```bash
wc example_1.html
```

### [bash] Как определить контрольную сумму MD5?

```bash
md5sum example_1.html
```

### [bash] Как определить контрольную сумму SHA1?

```bash
sha1sum example_1.html
```

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

### [bash] Как посмотреть настройки сетевой карты?

```bash
ip address show
```

### [bash] Как завершить процесс вежливо?

```bash
kill -15 123
```

### [bash] Как завершить процесс принудительно?

```bash
kill -9 123
```

### [bash] Как установить высокий приоритет процессу?

```bash
nice -n -19 sleep 30
```

### [bash] Как скачать файл или страницу?

```bash
curl -o example_1.html --user-agent "Bot" https://example.com/
```


### [bash] Как скачать файл или страницу? (второй вариант)

```bash
wget -O example_2.html -U "Bot" https://example.com/
```


### [bash] Как найти в файле подстроку?

```bash
grep -r "<title>" *
```


### [bash] Как искать файлы по расширению?

```bash
find ./ -iname "*.html"
```


### [bash] Как создать архив?

```bash
tar -cvzf example.tar.gz ./
```


### [bash] Как извлечь файлы из архива?

```bash
tar -xvzf example.tar.gz -C ./
```


### [bash] Как зашифровать файл?

```bash
gpg -c example.tar.gz
```


### [bash] Как расшифровать файл?

```bash
gpg -d example.tar.gz.gpg > example-copy.tar.gz
```


### [bash] Как загрузить файл на сервер по SSH?

```bash
scp doc.txt user@192.168.1.55:/home/developer/
```


### [bash] Как запустить встроенный в Python сервер?

```bash
python3 -m http.server 8084
```


### [bash] Как сменить владельца файла?

```bash
chown newuser example.txt
```


### [bash] Как разрешить чтение и запись только себе?

```bash
chmod 600 example.txt
```

### [bash] Как создать директорию?

```bash
mkdir example
```


### [bash] Как рекурсивно скопировать директорию?

```bash
cp -r ./example ./example_2
```


### [bash] Как посмотреть содержимое директории?

```bash
ls -ltrah
```


### [bash] Как записать текст в файл?

```bash
cat << EOF > example.txt
EXAMPLE
EOF
```

### [bash] Как в Podman выполнить очистку?

```bash
podman system prune --all --force
```

### [bash] Как в Podman развернуть и запустить СУБД?

```bash
podman run -dt --name devdb -e POSTGRES_PASSWORD=dev -e POSTGRES_USER=dev -e POSTGRES_DB=dev -p 5432:5432 postgres
```

### [bash] Как в Podman подключится к СУБД?

```bash
podman exec -it devdb psql -U dev -d dev
```


### [sql] Как узнать версию PostgreSQL?

```sql
SELECT VERSION();
```


### [sql] Как создать таблицу?

```sql
CREATE TABLE example (
    id SERIAL PRIMARY KEY,
    visitors INT
);
```


### [sql] Как добавить данные в таблицу?

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


### [sql] Как обновить запись?

```sql
UPDATE example SET visitors = 100 WHERE id = 8;
```


### [sql] Как удалить запись?

```sql
DELETE FROM example WHERE id = 8;
```


### [sql] Как узнать количество?

```sql
SELECT COUNT(*) FROM example;
```


### [sql] Как найти среднее значение?

```sql
SELECT AVG(visitors) FROM example;
```


### [sql] Как найти максимальное и минимальное значение?

```sql
SELECT MAX(visitors), MIN(visitors) FROM example;
```


### [sql] Как найти дисперсию?

```sql
SELECT VARIANCE(visitors) FROM example;
```


### [sql] Как найти сумму?

```sql
SELECT SUM(visitors) FROM example;
```


### [sql] Как найти ТОП-5 самых больших значений?

```sql
SELECT * FROM example ORDER BY visitors DESC LIMIT 5;
```


### [sql] Как найти все значения, которые находятся в нужном интервале?

```sql
SELECT * FROM example WHERE visitors BETWEEN 190 AND 800;
```


### [sql] Как найти дубликаты?

```sql
SELECT
    COUNT(visitors),
    visitors
FROM example
GROUP BY visitors
HAVING COUNT(visitors) > 1;
```


### [sql] Как узнать ранг?

```sql
SELECT
    id,
    visitors,
    dense_rank() OVER (ORDER BY visitors DESC) AS rank
FROM example;
```


### [sql] Как написать обобщённое табличное выражение?

```sql
WITH data AS (
    SELECT ARRAY[1, 2, 3, 4] AS alpha
)

SELECT * FROM data;
```


### [sql] Как сделать сочетание двух строк?

```sql
SELECT 26 AS age
UNION ALL
SELECT 40 AS age;
```


### [sql] Как написать свою функцию?

```sql
CREATE FUNCTION example_f(n integer)
RETURNS integer AS $$
BEGIN
    RETURN factorial(n);
END; $$
LANGUAGE plpgsql;
```


### [sql] Как узнать размер таблицы?

```sql
SELECT pg_size_pretty(pg_relation_size('example')) relation_size;
```

### [sql] Как запустить VACUUM?

```sql
VACUUM ANALYZE example;
```

### [sql] Как посмотреть количество страниц и кортежей?

```sql
SELECT relpages, reltuples FROM pg_class WHERE relname = 'example';
```


### [sql] Как создать индекс?

```sql
CREATE INDEX idx_example_visitors ON example(visitors);
```


### [sql] Как получить план выполнения запроса?

```sql
EXPLAIN (ANALYZE)
SELECT * FROM example WHERE id IN (1, 5, 7);
```


### [sql] Как скопировать таблицу?

```sql
CREATE TABLE backup_example AS TABLE example;
```


### [sql] Как использовать полнотекстовый поиск?

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


### [sql] Как найти данные с помощью регулярного выражения?

```sql
SELECT title, REGEXP_MATCHES(title, 'never') result FROM documents;
```


### [sql] Как найти длину строки и контрольную сумму?

```sql
SELECT title, LENGTH(title), MD5(title) FROM documents;
```


### [sql] Как написать функцию Root Mean Squared Error (RMSE)?

```sql
SELECT SQRT(AVG(POWER(beta - alpha, 2))) AS RMSE
FROM (
SELECT
unnest(ARRAY[8, 4, 9, 7, -9, 0, 2, 5, -8, -6, -4, -2, -1, 3]) as alpha,
unnest(ARRAY[0, -7, -5, -8, -1, -4, 1, 6, 5, 3, 8, -2, 7, -6]) as beta
);
```



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

### [python] Как воспользоваться классификатором?

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


X = df[["alpha", "beta"]]
target = df["target"]

model = RandomForestClassifier(n_estimators=1500, max_depth=200)
print(cross_val_score(model, X, target, cv=5, scoring="f1"))
```

### [python] Как выполнить нормализацию и стандартизацию данных?

```python
import numpy as np
from sklearn import preprocessing as pr


example = np.array([[20.], [30.], [10.], [50.]])

# [[-0.50709255], [0.16903085] ,[-1.18321596], [1.52127766]]
print(pr.StandardScaler().fit_transform(example))

# [[0.25], [0.5], [0.], [1.]]
print(pr.MinMaxScaler().fit_transform(example))

```

### [python] Как воспользоваться кластерным анализом KMeans?

```python
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=2).fit(df[["alpha", "beta"]])
print(kmeans.cluster_centers_)
print(kmeans.labels_)
```

### [python] Как воспользоваться кластерным анализом DBSCAN?

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


plt.style.use("bmh")
colors = np.array(["red", "green"])

iris = datasets.load_iris()
X = StandardScaler().fit_transform(iris.data[:, 2:4])

clusters = DBSCAN(eps=0.5, min_samples=3).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=colors[clusters])
plt.show()
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

### [python] Unit-test с неточной проверкой

```python
import unittest
from example import Example


class TestExample(unittest.TestCase):
    def test_resistance(self):
        result = Example().resistance(27, 3.14, 0.017)
        self.assertAlmostEqual(result, 0.146178343949, delta=1e-8)

if __name__ == "__main__":
    unittest.main()

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

<h2>GOLANG</h2>

### [go] Как найти расстояние между двумя точками в 2D?

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
    fmt.Println(math.Abs(result - 53.488) < 0.001)
}
```

### [go] Как написать поиск элемента в HashMap со сложностью O(1)?

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
    // Не очень хорошая идея с точки зрения аллокаций памяти
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


### [go] Как написать поиск элемента в Slice со сложностью O(n)?

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


### [go] Как написать поиск элемента в Slice со сложностью O(log n)?

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


### [go] Как написать функцию Root Mean Squared Error (RMSE)?

```go
func RootMeanSquareError(yTrue, yPred []float64) float64 {
    var sum float64 = 0
    for i := range yPred {
        sum += math.Pow(yTrue[i] - yPred[i], 2)
    }
    return math.Sqrt(sum / float64(len(yPred)))
}
```


### [go] Набор полезных математических функций

```go
package main

import (
    "math"
)

// Найти мощность зная вольты и амперы
func Power(volts, amps float64) float64 {
    return volts * amps
}

// Найти сопротивление зная вольты и амперы
func Resistance(volts, amps float64) float64 {
    return volts / amps
}

// Простой резистивный делитель
func VoltageDivider(volts, r1, r2 float64) float64 {
    return volts * (r2 / (r1 + r2))
}

// Закон Джоуля-Ленца
func JouleLenz(current, resistance, t float64) float64 {
    return resistance * (current*current) * t
}

// Сопротивление однородного проводника постоянного сечения
func ResistanceFromDimensions(length, area, resistivity float64) float64 {
    return resistivity * (length / area)
}

// Линейная скорость при известном периоде обращения
func LinearVelocity(radius, period float64) float64 {
    return (radius * (2 * math.Pi)) / period
}

// Путь при равноускоренном движении
func DistanceUniformAcceleration(velocity, acceleration, t float64) float64 {
    return (velocity * t) + ((acceleration * (t * t)) / 2)
}

// Ускорение при равноускоренном движении
func AccelerationUniformMotion(finalVelocity, initialVelocity, t float64) float64 {
    return (finalVelocity - initialVelocity) / t
}

// Кинетическая энергия
func KineticEnergy(mass, velocity float64) float64 {
    return 0.5 * mass * (velocity * velocity)
}

// Объём цилиндра
func CylinderVolume(radius, height float64) float64 {
    return math.Pi * (radius * radius) * height
}

// Объём сферы
func SphereVolume(radius float64) float64 {
    return (4.0 / 3.0) * math.Pi * (radius * radius * radius)
}

// Сигмоида для логистической регрессии
func Sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}
```

### [go] Простой минимальный unit-test с неточной проверкой

```go
package main

import (
    "math"
    "testing"
)

func TestExample(t *testing.T) {
    result := Sigmoid(0.1294)
    if math.Abs(result - 0.53230493545) > 1e-6 {
        t.Errorf("Sigmoid = %f", result)
    }
    result = SphereVolume(1.4571)
    if math.Abs(result - 12.9585582087) > 1e-6 {
        t.Errorf("SphereVolume = %f", result)
    }
    result = CylinderVolume(0.5475, 1.621)
    if math.Abs(result - 1.526515205278) > 1e-6 {
        t.Errorf("CylinderVolume = %f", result)
    }
    result = KineticEnergy(0.036, 141.2021)
    if math.Abs(result - 358.884595) > 1e-6 {
        t.Errorf("KineticEnergy = %f", result)
    }
    result = AccelerationUniformMotion(72, 0, 0.98)
    if math.Abs(result - 73.469388) > 1e-6 {
        t.Errorf("AccelerationUniformMotion = %f", result)
    }
    result = DistanceUniformAcceleration(72, 0.98, 3)
    if math.Abs(result - 220.41) > 1e-6 {
        t.Errorf("DistanceUniformAcceleration = %f", result)
    }
    result = LinearVelocity(5, 10)
    if math.Abs(result - 3.141593) > 1e-6 {
        t.Errorf("LinearVelocity = %f", result)
    }
    result = ResistanceFromDimensions(10, 2.01, 0.017)
    if math.Abs(result - 0.084577) > 1e-6 {
        t.Errorf("ResistanceFromDimensions = %f", result)
    }
    result = JouleLenz(9.41, 145, 4)
    if math.Abs(result - 51357.898) > 1e-6 {
        t.Errorf("JouleLenz = %f", result)
    }
    result = VoltageDivider(12, 450, 231)
    if math.Abs(result - 4.07048458) > 1e-6 {
        t.Errorf("VoltageDivider = %f", result)
    }
    result = Resistance(12, 2)
    if math.Abs(result - 6) > 1e-6 {
        t.Errorf("Resistance = %f", result)
    }
    result = Power(12, 2)
    if math.Abs(result - 24) > 1e-6 {
        t.Errorf("Power = %f", result)
    }
}
```

### [go] Как умножить две квадратные матрицы?

```go
package main

import (
    "fmt"
)

func main() {
    a := [][]uint8{
        {1, 0},
        {0, 2},
    }
    b := [][]uint8{
        {10, 1},
        {1, 20},
    }
    result := [][]uint8{
        {0, 0},
        {0, 0},
    }
    //  Сложность алгоритма O(n^3)
    for i := 0; i < len(a); i++ {
        for j := 0; j < len(b[0]); j++ {
            for k := 0; k < len(a[0]); k++ {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    fmt.Println(result)
}
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

### [go] Чтение данных из файла JSON

```go
package main

import (
    "os"
    "io/ioutil"
    "encoding/json"
)

type Url struct {
    ID int `json:"id"`
    Title string `json:"title"`
    URL string `json:"url"`
}

func ReadUrlsFromFile(filename string) ([]Url, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    bytes, err := ioutil.ReadAll(file)
    if err != nil {
        return nil, err
    }

    var urls []Url
    err = json.Unmarshal(bytes, &urls)
    if err != nil {
        return nil, err
    }

    return urls, nil
}
```


