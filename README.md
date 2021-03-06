# C2monac análisis estadístico de mercados financieros

C2monac es un soft en Python 3 para análisis de mercados financieros. Por el momento soporta solo acciones el mercado merval de Argentina y bitcoins (BTC) pero en el futuro agregará soporte para otros mercados.

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

### Estimación del valor de pampa a 30 días por Montecarlo usando datos entre fechas:

```
>>> import c2monac.sim as sm
>>> sim = sm.montecarlo('pamp',30,'02/28/17','08/28/17')
>>> sm.analiza_resultados(sim)
('normal', (46.380645509577988, 4.2302062787977972))
```

### Estimación del valor de pampa a 30 días por Montecarlo con bootstraping usando datos entre fechas:

```
>>> import c2monac.sim as sm
>>> sim = sm.montecarlobs('pamp',30,'02/28/17','08/28/17')
>>> sm.analiza_resultados(sim)
('normal', (45.873440000000002, 4.3650124359960305))
```

### Gráfico de correlación entre índice merval y pamp:

```
>>> import c2monac.misc as misc
>>> misc.mervcorr('pamp')
```

La imagen generada se puede ver en este [link](http://10mp.net/gitimg/pampco.png)

### Cálculo de riesgo de cartera por Markowitz

```
>>> from c2monac.stockdefs import *
>>> val = Stock()
>>> val.papel = 'pamp'
>>> pamp = val.trae_datos()
>>> val.papel = 'agro'
>>> agro = val.trae_datos()
>>> bma = val.trae_datos()
>>> val.papel = 'bma'
>>> bma = val.trae_datos()
>>> import c2monac.misc as msc
>>> vpamp = msc.variaciones_diarias(pamp)
>>> vagro = msc.variaciones_diarias(agro)
>>> vbma = msc.variaciones_diarias(bma)
>>> msc.markowitz( (vpamp,vagro,vbma), (.5,.25,.25))
(matrix([[ 0.14346939]]), matrix([[ 0.93245608]]))
```

### Estimación del valor del bitcoin a 42 días por Montecarlo:

```
>>> import c2monac.sim as sm
>>> sim = sm.montecarlo_btc()
>>> sm.analiza_resultados(sim)
('normal', (4559.5009531976284, 397.88022695597328))
```

### Estimación del valor del bitcoin a 42 días por Montecarlo entre fechas:

```
>>> import c2monac.sim as sm
>>> sim = sm.montecarlo_btc(dias=21,desde='2017-02-28',hasta='2017-08-28')
>>> dist, params = sm.analiza_resultados(sim)
>>> dist
'johnsonsu'
>>> params
(0.19352807920786852, 2.7513541601261418, 4826.8132852245153, 1318.2029555337162)
>>> import scipy.stats as stats
>>> stats.johnsonsu.mean(params[0],params[1],params[2],params[3])
4727.6791419796873
>>> stats.johnsonsu.std(params[0],params[1],params[2],params[3])
513.92115572690886
>>> stats.johnsonsu.pdf( 4727.6791, params[0], params[1], params[2], params[3])
0.00083025567622726105
```
