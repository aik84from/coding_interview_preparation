# Coding interview preparation

Часто задаваемые вопросы: https://aik84from.github.io/faq.html

(C) Калинин Александр Игоревич

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

