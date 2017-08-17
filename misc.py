import warnings
import subprocess
import numpy as np
import multiprocessing
from scipy import stats
from .stockdefs import *
import matplotlib.pyplot as plt



#Calcula cantidad de clases del histograma
def cant_clases_hist(self,cant_datos):
    """Calculo de clases de histograma"""

    if cant_datos > 1000:
        return 31  
    if cant_datos < 1000 and cant_datos > 100:
        return 15
    if cant_datos < 100:
        return 11


#obtiene lista de cierres del pandas data object
def cierre(self,pddatos):
    """Obtiene valor de cierre desde dato tipo pandas"""
    return  [ i for i in pddatos['Close'] ]


#calcula variaciones diarias del papel
def variaciones_diarias(self, pddatos):
    """Calcula variaciones diarias del papel"""
    a = self.cierre(pddatos)
    return  [i for i in map(lambda x,y:y-x, a[:-1], a[1:]) ]


#calcula distribucion mejor distribucion para un histograma
def best_fit_distribution(self,data, bins=15, ax=None):
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
                    
                r.append(pool.apply_async(self.fit_by_core, args=[data, distribution,x,y]))
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
def fit_by_core(self,data,distribution,x,y):
    """Procesa en cada core la distribucion de probabilidad"""
        
    params = distribution.fit(data)

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Calculate fitted PDF and error with fit in distribution
    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
    sse = np.sum(np.power(y - pdf, 2.0))

    return sse, distribution, params


#calcula media y std para analisis de riesgo
def riesgo(self,papel,v=None):
    """calcula riesgo de papel"""
    val = stockdefs.Stock()
    val.papel = papel
    val.vebose = v

    datos = val.trae_datos()
#    if v is not None:
#        print("Datos Obtenidos")
#        print("Verificando Distribucion")

    b = self.variaciones_diarias(datos)
    analiza_resultados(b,v=1)


#analisis de riesgo de cartera por Markowitz
def markowitz(self,papeles, pesos):
    """Calculo de riesgo de cartera por Markowitz"""

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
def mervcorr(self,papel):
    """Calcula y grafica correlacion de papel y merval"""

    import matplotlib.patches as mpatches

    val = stockdefs.Stock()
    val.papel = 'iar'

    merv = val.trae_datos()
    a = self.cierre(merv)

    val.papel = papel
    dpapel = val.trae_datos()
    b = self.cierre(dpapel)

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
def mvarcorr(self,papel):
    """Calcula y grafica correlacion de variaciones diarias 
    entre papel y merval"""

    import matplotlib.patches as mpatches

    val = stockdefs.Stock()
    val.papel = 'iar'

    merv = val.trae_datos()
    a = self.variaciones_diarias(merv)

    val.papel = papel
    dpapel = val.trae_datos()
    b = self.variaciones_diarias(dpapel)

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



def comparacion(self,*papeles, desde=None):
    """Grafica comparacion de papeles"""

    val = stockdefs.Stock()

    if desde is not None:
        val.desde = desde

    datos = []
    for i in papeles:
        val.papel = i
        x = val.trae_datos()
        plt.plot(cierre(x))
        
    plt.grid(True)
    plt.show()
