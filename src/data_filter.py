from os import name
import wfdb;
import numpy as np;
import matplotlib.pyplot as plt;
from data_loader import Loader as data;
#Objeto wfdb, aplica función dl_database para descargar la base de datos 'mitdb'
#función rdrecord busca el registro de la ruta dada, y lo almacena en record, que es un objeto de la clase Record. 
# Este objeto contiene información sobre la señal, como el número de canales, la frecuencia de muestreo, 
# el número de muestras, etc.
class Filter():

    def __init__(self, record_path) -> None:
        loaderr = data(record_path)
        self.record_path = str(record_path)
        self.record = loaderr.record
        self.annotation = loaderr.annotation
        print("Record:",self.record," Annotation:",self.annotation)
        pass

    def bandpass_filter(self, signaldata, lowcut, highcut, fs, order):
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        y = filtfilt(b, a, signaldata)
        return y

if __name__ == "__main__":
    ecg = data('../src/data/mitdb/100')
    filtro = Filter(ecg.record_path)

    #Creamos una señal cualquiera, sabemos que la frecuencia de muestreo es de 360 Hz
    #La frecuencia de la señal es de 30 Hz, nos servirá para probar el filtro
    fm = 1.2 #Frecuencia del latido
    fs = 360 #Frecuencia de muestreo
    duración = 10
    x = np.linspace(0, duración, int(fs * duración))
    latido = np.sin(2 * np.pi * fm * x)
    plt.figure(figsize=(10, 6))
    plt.title("Señal Artificial Original")
    plt.plot(x, latido)
    plt.xlim(0, 10)
    plt.show()
    #Pero esta señal aún no tiene ruido, por lo que primero le añadiremos ruido de forma aleatoria para 
    #Simular una señal real, y luego aplicaremos el filtro para ver si es capaz de eliminar el ruido.
    ruido = np.random.normal(0, 0.5, latido.shape)
    latido_ruidoso = latido + ruido
    plt.plot(x, latido_ruidoso)
    plt.title("Señal Artificial Ruidosa")
    plt.show()
    #Ya tenemos la señal con ruido, ahora aplicamos el filtro para ver si es capaz de eliminar el ruido.
    latido_filtrado = filtro.bandpass_filter(latido_ruidoso, 0.5, 15, fs, order=5) 
    #Usamos de Frecuencia de corte alta de 15 Hz ya que es una señal sintética
    #Para señales reales, se puede usar una frecuencia de corte alta de 40 Hz, ya que la mayoría de las señales de ECG tienen frecuencias menores a 40 Hz.
    plt.plot(latido_filtrado)
    plt.title("Señal Artificial filtrada")
    plt.show()
    #Podemos ver que nuestra señal filtrada es muy similar a la señal original, con menos ruido.
    #También podemos decir que nuestro filtro es capaz de eliminar el ruido de la señal.
    #Pero, ahora tenemos que aplicar el filtro a una señal real, para ver si es capaz de eliminar el ruido de la señal.

    final = int(5 * ecg.record.fs)  # 10 segundos de datos
    time = np.arange(0, final) / ecg.record.fs
    seg = ecg.record.p_signal[:final, 0]  # Tomamos solo el primer canal
    plt.plot(seg)
    plt.title("Latido")
    plt.show()
    reg_ruidosa = seg + ruido[0:final]
    plt.plot(reg_ruidosa)
    plt.xlim(0, final)
    plt.title("Latido Ruidoso")
    plt.show()
    #Aquí ya vemos que la señal tiene un poco de ruido, pero no es mucho, por lo que aplicaremos el filtro para ver si es capaz de eliminar el ruido.
    seg_filtrada = filtro.bandpass_filter(seg, 0.5, 40, ecg.record.fs, order=5)
    plt.plot(seg_filtrada)
    plt.title("Latido filtrado")
    plt.show()
    #Nótese que la señal filtrada es muy similar a la señal original, con menos ruido. Esto nos indica que nuestro filtro es capaz de eliminar el ruido de la señal.
    #Pero, podemos medir esto?, si, utilizaremos np.corrcoef para medir la correlación entre la señal original y la señal filtrada, 
    #Si el valor es cercano a 1, significa que las señales son muy similares.
    correlacion = np.corrcoef(seg, seg_filtrada)[0, 1]
    print(f'Correlación entre la señal original y la señal filtrada:' , correlacion)
    #Haremos lo mismo con la señal sintética, para ver si el filtro es capaz de eliminar el ruido de la señal sintética.
    correlacion_sintetica = np.corrcoef(latido, latido_filtrado)[0, 1]
    print(f'Correlación entre la señal sintética y la señal filtrada:' , correlacion_sintetica)
    #Ahora, 