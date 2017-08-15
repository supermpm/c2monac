import datetime
import pandas_datareader.data as web


#Clase para procesos basicos de acciones
class Stock():
    """Tipo de datos basico para calculo estadistico de acciones"""
    
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
        """Obtiene datos desde google finance"""
        papel = 'bcba:' + self.papel
        datos = web.DataReader(papel, data_source='google', start=self.desde, end=self.hasta)
        self.ultimo_cierre = datos.ix[-1]['Close']
        return datos

