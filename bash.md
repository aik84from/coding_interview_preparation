# Coding interview preparation

Часто задаваемые вопросы: https://aik84from.github.io/faq.html

(C) Калинин Александр Игоревич


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


### [bash] Как узнать свободное место на диске?

```bash
df -h
```


### [bash] Как посмотреть память? (свободное место)

```bash
free -h
```


### [bash] Как посмотреть процессы?

```bash
ps -A
```


### [bash] Как загрузить файл?

```bash
curl -o example_1.html --user-agent "Bot" https://example.com/
```


### [bash] Как загрузить файл?

```bash
wget -O example_2.html -U "Bot" https://example.com/
```


### [bash] Как посмотреть содержимое файла?

```bash
cat example_1.html | less
```


### [bash] Как увидеть первые 5 строк?

```bash
head 5 example_1.html
```


### [bash] Как увидеть крайние 5 строк?

```bash
tail 5 example_1.html
```


### [bash] Как посчитать количество строк, слов и байт?

```bash
wc example_1.html
```


### [bash] Как определить контрольную сумму?

```bash
md5sum example_1.html
```


### [bash] Как определить контрольную сумму?

```bash
sha1sum example_1.html
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



