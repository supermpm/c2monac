# Análisis estadístico de mercados financieros

C2monac es un soft en Python 3 para análisis de mercados financieros. Por el momento soporta solo acciones el mercado merval de Argentina pero en las próximas versiones agregara soporte para otros mercados y bitcoins.

## Requerimientos:

- python 3
- pandas
- scipy
- numpy
- Se recomienda usar Anaconda 3
- Si bien no tiene requerimientos específicos de hardware se recomienda como mínimo procesador Core i5 de 4 cores ya que las simulaciones pueden tomarse varios minutos en correr. Cuantos más cores y threads posea el procesador menor sera el tiempo de ejecución.



### Instalacion:

```
$ git clone https://github.com/supermpm/c2monac
```

## Ejemplos:

### Estimacion del valor de pampa a 42 dias por Montecarlo:

```
import c2monac.sim as sm
sm.montecarlo('pamp')
```

### Grafico de correlacion entre indice merval y pamp:

```
import c2monac.misc as misc
misc.mervcorr('pamp')
```


