# Coding interview preparation

Часто задаваемые вопросы: https://aik84from.github.io/faq.html

(C) Калинин Александр Игоревич


### [html] Как написать инструкцию в формате HTML?

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<link rel="shortcut icon" type="image/x-icon" href="/favicon.png" />
<link href="readme.css" type="text/css" rel="stylesheet" />
<title>Руководство пользователя</title>
</head>
<body>
<div class="container">

<h1>Руководство пользователя</h1>
<h2>Заголовок</h2>
<p>Полезная инструкция и <a href="https://aik84from.github.io/" target="_blank">ссылка</a> на сайт.</p>
<h3>Пример кода</h3>
<pre><code>curl https://example.com/</code></pre>

</div>
</body>
</html>
```


### [css] Как создать CSS для простой инструкции?

```css
body {
  font-family: Arial, sans-serif;
}

h1 {
  font-size: 48px;
  font-weight: 400;
}

h2 {
  font-size: 36px;
  font-weight: 200;
}

h3 {
  font-size: 20px;
  font-weight: 200;
}

p {
  font-size: 16px;
  line-height: 1.5;
}

a {
  color: #211180;
}

pre {
  color: #adacb5;
  background-color: #282630;
  border-left: 8px solid #43424a;
  padding: 8px;
  margin: 0;
  font-family: monospace;
  overflow-x: auto;
  overflow-y: hidden;
}

.container {
  background-color: #FFF;
  padding: 4px;
  margin: 0;
}

@media screen and (min-width: 1200px) {
  .container {
    width: 1000px;
    padding: 4px;
    margin: 0 auto;
  }
}
```


