# C2monac análisis estadístico de mercados financieros

C2monac es un soft en Python 3 para análisis de mercados financieros. Por el momento soporta solo acciones el mercado merval de Argentina pero en el futuro agregará soporte para otros mercados y bitcoins.

## Requerimientos:

- python 3
- pandas
- scipy
- numpy
- Se recomienda usar Anaconda 3
- Si bien no tiene requerimientos específicos de hardware se recomienda como mínimo procesador Core i5 de 4 cores ya que las simulaciones pueden tomarse varios minutos en correr. Cuantos más cores y threads posea el procesador menor sera el tiempo de ejecución.



### Instalación:

```
$ git clone https://github.com/supermpm/c2monac
```

## Ejemplos:

### Estimación del valor de pampa a 42 días por Montecarlo:

```
>>> import c2monac.sim as sm
>>> sim = sm.montecarlo('pamp')
>>> sm.analiza_resultados(sim)
('normal', (46.653857212320446, 4.1409944487857775))
```

### Gráfico de correlación entre índice merval y pamp:

```
>>> import c2monac.misc as misc
>>> misc.mervcorr('pamp')
```

La imagen generada se puede ver en este (link)[http://10mp.net/gitimg/pampco.png]
