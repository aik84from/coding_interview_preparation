# Coding interview preparation

Часто задаваемые вопросы: https://aik84from.github.io/faq.html

(C) Калинин Александр Игоревич


### [javascript] Как написать простую библиотеку для закона Ома?

```javascript
const volts = (amperes, ohms) => amperes * ohms;

const amperes = (volts, ohms) => volts / ohms;

const ohms = (volts, amperes) => volts / amperes;

const watts = (volts, amperes) => volts * amperes;

const voltageDivider = (volts, r1, r2) => volts * (r2 / (r1 + r2));


module.exports = {
    volts: volts,
    amperes: amperes,
    ohms: ohms,
    watts: watts,
    voltageDivider: voltageDivider
};
```


