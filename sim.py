from .misc import *
from .stockdefs import *
#import .stockdefs as st
import functools
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


#calulo valor a futuro por montecarlo
def montecarlo(papel,dias=None,v=None):
    """Calulo valor a futuro por montecarlo"""

    val = Stock()
#    M = Cbas()
    val.papel = papel
    if dias is None:
        dias = val.periodo
    datos = val.trae_datos()

    clases_hist = cant_clases_hist(len(cierre(datos)))

    b = variaciones_diarias(datos)

    dist, dparams = best_fit_distribution(b, bins = clases_hist)


    f40d = []
    #Genera objeto stats."distribucion" para no tener que hacer un if
    #por cada distribucion, por ejemplo scipy.stats.norm
    ob = eval('stats.'+dist)

    for x in range(0,5000):
        s = ob.rvs(*dparams, size=dias)
        f = functools.reduce(lambda x,y: x+y, s) + val.ultimo_cierre
        f40d.append(float(f))

    return f40d

#calulo valor a futuro por montecarlo bootstraping
def montecarlobs(papel,dias=None):
    """Calulo valor a futuro por montecarlo bootstraping"""

#    val = stockdefs.Stock()
    val = Stock()
#    M = Cbas()
    val.papel = papel
    if dias is None:
        dias = val.periodo

    datos = val.trae_datos()

    b = variaciones_diarias(datos)


    f40d = []
    for x in range(0,5000):
        rnd = np.random.randint(0,len(b), size=dias)
        k = []
        for i in rnd:
            k.append(b[i])

        f = functools.reduce(lambda x,y: x+y, k) + val.ultimo_cierre
        f40d.append(float(f))


    return f40d

#analisis de resultados de simulacion de montecarlo
def analiza_resultados(resultados):
    """Analisis de resultados de simulaciones"""

#    M = Cbas()

    normstat, pvalue = stats.mstats.normaltest(resultados)
        
    if pvalue > .055:
        rmean = np.mean(resultados) 
        rstd = np.std(resultados)

        return ('normal',(rmean,rstd))
        
        
    else: 

        clases_hist = cant_clases_hist(len(resultados))
        rdist, rparams = best_fit_distribution( resultados, bins = clases_hist)


        return (rdist, rparams)
            
#calculo a futuro de carteta por montecarlo
def simula_cartera(*resultados):
    """Simulacion de cartera"""
    a = []
    for j in range(0, len(resultados[0])):
        suma = 0
        for k in range(0,len(resultados)):
            suma += resultados[k][j]
        a.append(suma) 

    return a
