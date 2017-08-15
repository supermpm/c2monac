import sys
import datetime
import warnings
import functools
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from scipy import stats
#import matplotlib.rcparams
import matplotlib.pyplot as plt
import pandas_datareader.data as web


#Clase para procesos basicos de acciones
class Stock():
    
    def __init__(self):
        fecha = datetime.datetime.now()
        anio_anterior = int(fecha.strftime("%Y"))-1
        fecha_anterior = fecha.replace(year=anio_anterior)
        
        self.hasta = fecha.strftime("%m/%d/%Y")
        self.desde = fecha_anterior.strftime("%m/%d/%Y")
        self.papel = 'iar'
        self.periodo = 42
        self.precio_compra = None
        self.verbose = None
        self.ultimo_cierre = 0
	
    #obtiene datos de acciones de google finance
    def trae_datos(self): 
        papel = 'bcba:' + self.papel
        datos = web.DataReader(papel, data_source='google', start=self.desde, end=self.hasta)
        self.ultimo_cierre = datos.ix[-1]['Close']
        return datos


#Calcula cantidad de clases del histograma
def __cant_clases_hist(cant_datos):

    if cant_datos > 1000:
        return 31  
    if cant_datos < 1000 and cant_datos > 100:
        return 15
    if cant_datos < 100:
        return 11


#calcula variaciones diarias del papel
def variaciones_diarias(pddatos):
    a = cierre(pddatos)
    return  [i for i in map(lambda x,y:y-x, a[:-1], a[1:]) ]


#obtiene lista de cierres del pandas data object
def cierre(pddatos):
    return  [ i for i in pddatos['Close'] ]


#calcula distribucion mejor distribucion para un histograma
def best_fit_distribution(data, bins=15, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
        
    # Distributions to check
    DISTRIBUTIONS = [        
        stats.alpha,stats.anglit,stats.arcsine,stats.beta,stats.betaprime,stats.bradford,stats.burr,
        stats.cauchy,stats.chi,stats.chi2,stats.cosine, stats.dgamma,stats.dweibull,stats.erlang,
        stats.expon,stats.exponnorm,stats.exponweib,stats.exponpow,stats.f,stats.fatiguelife,stats.fisk,
        stats.foldcauchy,stats.foldnorm,stats.frechet_r,stats.frechet_l,stats.genlogistic,stats.genpareto,
        stats.gennorm,stats.genexpon,stats.genextreme,stats.gausshyper,stats.gamma,stats.gengamma,
        stats.genhalflogistic,stats.gilbrat,stats.gompertz,stats.gumbel_r, stats.gumbel_l,stats.halfcauchy,
        stats.halflogistic,stats.halfnorm,stats.halfgennorm,stats.hypsecant,stats.invgamma,stats.invgauss,
        stats.invweibull,stats.johnsonsb,stats.johnsonsu,stats.ksone,stats.laplace,stats.logistic,
        stats.loggamma,stats.loglaplace,stats.lognorm,stats.lomax,stats.maxwell,stats.mielke,stats.nakagami,
        stats.ncx2,stats.ncf,stats.nct,stats.norm,stats.pareto,stats.pearson3,stats.powerlaw,stats.powerlognorm,
        stats.powernorm,stats.rdist,stats.reciprocal, stats.rayleigh,stats.rice,stats.recipinvgauss,
        stats.semicircular,stats.t,stats.triang,stats.truncexpon,stats.truncnorm,stats.tukeylambda,
        stats.uniform,stats.vonmises,stats.vonmises_line,stats.wald,stats.weibull_min,
        stats.weibull_max,stats.wrapcauchy
    ]
        
    # Best holders
    best_distribution = stats.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    
    # Estimate distribution parameters from data
    r = []
    for distribution in DISTRIBUTIONS:
    
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
    
                count = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(processes=count)
                    
                r.append(pool.apply_async(fit_by_core, args=[data, distribution,x,y]))
                pool.close()
                pool.join()
    
    
        except Exception:
            pass
    
    for k in r:
        if best_sse > k.get()[0] > 0:
            best_distribution = k.get()[1]
            best_params = k.get()[2]
            best_sse = k.get()[0]
    
    return (best_distribution.name, best_params)
    
        
        
#manda calculo de distribuciones a todos los cores
def fit_by_core(data,distribution,x,y):
        
    
    params = distribution.fit(data)

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Calculate fitted PDF and error with fit in distribution
    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
    sse = np.sum(np.power(y - pdf, 2.0))

    return sse, distribution, params


#calulo valor a futuro por montecarlo
def simulacion(papel,dias=None,v=None):

    val = Stock()
    val.papel = papel
    if dias is None:
        dias = val.periodo
    datos = val.trae_datos()
    if v is not None:
        val.verbose = 1

    if val.verbose is not None:
        print("Datos Obtenidos")
        print("Verificando Distribucion")

    clases_hist = __cant_clases_hist(len(cierre(datos)))

    b = variaciones_diarias(datos)

    dist, dparams = best_fit_distribution(b, bins = clases_hist)

    if val.verbose is not None:
        print("\n")
        print("Distribucion Movimientos Diarios:",dist+", Params:",dparams)
        print("Comienzo Simulacion")



    f40d = []
    #Genera objeto stats."distribucion" para no tener que hacer un if
    #por cada distribucion, por ejemplo scipy.stats.norm
    ob = eval('stats.'+dist)

    for x in range(0,5000):
        s = ob.rvs(*dparams, size=dias)
        f = functools.reduce(lambda x,y: x+y, s) + val.ultimo_cierre
        f40d.append(float(f))

    if val.verbose is not None:
        print("Fin simulacion")

#        if val.verbose is not None:
        plt.hist(b, 15)
        plt.grid(True)
        plt.show()

    return f40d

    #calulo valor a futuro por montecarlo bootstraping
def montecarlobs(papel,dias=None):

    val = Stock()
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
def analiza_resultados(resultados,v=None):


        normstat, pvalue = stats.mstats.normaltest(resultados)
        
        if pvalue > .055:
#            if self.verbose is not None:
            if v is not None:
                print("\n")
                print("pvalor:", str(pvalue))
        
            rmean = np.mean(resultados) 
            rstd = np.std(resultados)

            return ('normal',(rmean,rstd))
        
        
        else: 

            clases_hist = __cant_clases_hist(len(resultados))
            rdist, rparams = best_fit_distribution( resultados, bins = clases_hist)

            
            #Genera objeto stats."distribucion" para no tener que hacer un if
            #por cada distribucion, por ejemplo scipy.stats.norm
#            ob = eval('stats.' + rdist)


            return (rdist, rparams)
            
#calculo a futuro de carteta por montecarlo
def simula_cartera(*resultados):
    a = []
    for j in range(0, len(resultados[0])):
        suma = 0
        for k in range(0,len(resultados)):
            suma += resultados[k][j]
        a.append(suma) 

    return a

#calcula media y std para analisis de riesgo
def riesgo(papel,v=None):
    val = Stock()
    val.papel = papel
    val.vebose = v

    datos = val.trae_datos()
    if v is not None:
        print("Datos Obtenidos")
        print("Verificando Distribucion")

    b = variaciones_diarias(datos)
    analiza_resultados(b,v=1)


#analisis de riesgo de cartera por Markowitz
def markowitz(papeles, pesos):

    varmat = np.asmatrix(papeles)


    p = np.asmatrix(np.mean(varmat, axis=1))
    #ATENCION!!! Aca hago la transpuesta porque necesito que quede todo en la misma fila
    p = p.T
    w = np.asmatrix(pesos)
    C = np.asmatrix(np.cov(varmat))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    return (mu, sigma)

#grafica correlacion de valores de accion vs merv
def mervcorr(papel):

    import matplotlib.patches as mpatches

    val = Stock()
    val.papel = 'iar'

    merv = val.trae_datos()
    a = cierre(merv)

    val.papel = papel
    dpapel = val.trae_datos()
    b = cierre(dpapel)

    r = stats.linregress(a,b)
    r_params = "r="+str(r[2])+" p="+str(r[3])

    reg = [ r[1]+i*r[0] for i in a ]


    plt.plot(a, reg, 'r')
    plt.plot(a,b,'.')
    plt.xlabel('merv')
    plt.ylabel(papel)
    red_patch = mpatches.Patch(color='red', label=r_params)
    plt.legend(handles=[red_patch])
    plt.grid(True)
    plt.show()


#grafica correlacion de variaciones diarias de accion vs merv
def mvarcorr(papel):

    import matplotlib.patches as mpatches

    val = Stock()
    val.papel = 'iar'

    merv = val.trae_datos()
    a = variaciones_diarias(merv)

    val.papel = papel
    dpapel = val.trae_datos()
    b = variaciones_diarias(dpapel)

    r = stats.linregress(a,b)
    r_params = "r="+str(r[2])+" p="+str(r[3])

    reg = [ r[1]+i*r[0] for i in a ]


    ma = me = 0
    for i in b:
        ma+=1 if i>=0 else 0
        me+=1 if i<0 else 0

    plt.plot(a,reg,"r")
    plt.plot(a,b,'.')
    plt.xlabel('merv')
    plt.ylabel(papel)
    red_patch = mpatches.Patch(color='red', label=r_params)
    plt.legend(handles=[red_patch])
    plt.grid(True)
    plt.show()
    print("Proporcion mayor a 0:",ma / len(b))
    print("Proporcion menor a 0:",me / len(b))
    print("Relacion:",(ma/me)-1 )



def comparacion(*papeles,desde=None):

    val = Stock()

    if desde is not None:
        val.desde = desde

    datos = []
    for i in papeles:
        val.papel = i
        x = val.trae_datos()
        plt.plot(cierre(x))
        
    plt.grid(True)
    plt.show()
        

