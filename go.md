# Coding interview preparation

Часто задаваемые вопросы: https://aik84from.github.io/faq.html

(C) Калинин Александр Игоревич


### [go] Как найти расстояние между двумя точками на карте?

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


### [go] Как написать формулу сигмоиды для логистической регрессии?

```go
func sigmoid(x float64) float64 {
    return 1.0 / (1 + math.Exp(-x))
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


### [go] Как выполнить мониторинг состояния жидкостного охлаждения?

```go
// Вымышленный схематический пример системы
// мониторинга жидкостного охлаждения сервера
func main() {
    for {
        if leakSensor() > threshold {
            activateSiren("Утечка охлаждающей жидкости")
        }
        if thermoResistor() > threshold {
            activateSiren("Сервер сильно нагрелся")
        }
        if pressureSensor() < threshold {
            activateSiren("Упало давление охлаждающей жидкости")
        }
        if potentiometer() < threshold {
            activateSiren("Закрыт люк системы охлаждения")
        }
        if photoResistor() < threshold {
            activateSiren("Слишком мутная жидкость охлаждения")
        }
        time.Sleep(1 * time.Second)
    }
}
```


### [go] Как создать математический микросервис (экономика, анализ датчиков и т.д.)?

```go
package main

import (
	"encoding/json"
	"fmt"
)

// Прежде всего, определяется контракт данных.
type RequestData struct {
	Example []float64 `json:"example"`
}

// Далее формируются правила валидации.
// В некоторых проектах добавляют целый набор
// экспертных правил на этап валидации.
// Или даже подключают модели, например,
// логистическую регрессию или простые деревья.
func Validate(arr []float64) error {
	if len(arr) < 5 {
		return fmt.Errorf("len(arr) < 5 (len = %d)", len(arr))
	}
	for _, x := range arr {
		if x < -100 || x > 100 {
			return fmt.Errorf("x < -100 || x > 100 (x = %f)", x)
		}
	}
	return nil
}

// Набор формул (ядро принятия решений).
// Как правило, самая большая и сложная часть.
// Часто оформляют как библиотеку с набором функций.
// В реальных системах математические модули пишут
// через TDD (test-driven development) с очень
// подробным набором unit-тестов.
func Example(arr []float64) []float64 {
	result := make([]float64, len(arr))
	for i, v := range arr {
		result[i] = v * v
	}
	return result
}

// Вместо этого кода ваш шаблон (скелет) микросервиса для расчётов.
func main() {
	var raw []byte = []byte(`{"example": [1, 2, 4, 50, -54, -99]}`)
	var data RequestData
	err := json.Unmarshal(raw, &data)
	if err != nil {
		fmt.Println("JSON Decode", err)
		return
	}
	err = Validate(data.Example)
	if err != nil {
		fmt.Println("Validate", err)
		return
	}
	fmt.Println("result", Example(data.Example))
}
```

