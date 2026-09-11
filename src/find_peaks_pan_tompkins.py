#Cómo introducción, el proceso de Pan-Tompkins es un algoritmo para la detección de picos R en señales de ECG. 
#Este algoritmo se basa en la idea de que los picos R son los puntos más altos de la señal de ECG, y por lo tanto,
#se pueden detectar mediante un proceso de filtrado y umbralización.

#Este proceso implica algunos pasos:
#1.-Obtener la tasa de crecimiento, ya que sabiendo que los picos R son los puntos más altos de la señal,
#Podemos obtener la tasa de crecimiento de la señal, y así obtener los picos R.
#2.-Elevar todo al cuadrado, de esta manera, los picos R serán los puntos más altos de la señal, y los picos negativos serán los puntos más bajos de la señal.
#3.-Aplicar un filtro de media móvil, de esta manera, los picos R serán los puntos más altos de la señal, y los picos negativos serán los puntos más bajos 
#4.-Buscar los picos R en la señal ya suavizada
from pathlib import Path;
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.signal import find_peaks;
from data_loader import Loader as data;
from data_filter import Filter as filtro;
class Pan_tompkins():

    def __init__(self, record_path):
        loaderr = data(record_path)
        self.record_path = str(record_path)
        self.record = loaderr.record
        self.annotation = loaderr.annotation
        #print("Record:",self.record," Annotation:",self.annotation)
        pass

    def find_peaks(self, record, graficar=False, duracion=None):
        if duracion == None:
            n_muestras = len(record.p_signal)
        else:
            n_muestras = int(duracion * record.fs)
        time = np.arange(0, n_muestras) / record.fs
        segn = record.p_signal[:n_muestras, 0] 
        #Ahora, tenemos que filtrar la señal previo a derivarla, para tener una señal más limpia
        seg = filtro.bandpass_filter(segn, 0.5, 40, record.fs, order=5)
        #Ahora, que obtuvimos la señal, debemos encontrar su "derivada" o tasa de crecimiento, para esto, podemos usar la función np.diff
        derivada = np.diff(seg)
        #Ahora, debemos elevar todo al cuadrado, de esta manera, los picos R serán los puntos más altos de la señal, y los picos negativos serán los puntos más bajos de la señal.
        record_squared = derivada ** 2
        #Ahora, aplicaremos la convolución, de este modo, podremos suavizar la señal, y así poder encontrar los picos R. 
        #Para esto, podemos usar la función np.convolve, y un filtro de media móvil de 150 ms.
        convolution_window = int(0.150 * record.fs)  # 150 ms
        window = np.ones(convolution_window) / convolution_window
        convolv = np.convolve(record_squared, window, mode='same')
        #Aquí vemos que la señal ya está suavizada, y podemos ver los picos R, pero aún no hemos encontrado los picos R
        #Ahora, debemos buscar los picos R en la señal ya suavizada, para esto, podemos usar la función find_peaks de scipy.signal.find_peaks, y un umbral de 0.5
        umbral = 0.3 * convolv.max()
        picos = find_peaks(convolv, height = umbral, distance = int(0.25 * record.fs))[0]
        picos = self.refinar_picos(picos, seg, record.fs)
        if graficar == True:
            plt.plot(time,seg)
            plt.scatter(time[:-1][picos], seg[:-1][picos], color='red', label='R-peaks')
            plt.title('Señal con los picos R detectados')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.show()
        return picos, time, seg, convolv
    @staticmethod
    def refinar_picos(picos_candidatos, seg, record_fs, ventana_atras_ms=100, ventana_adelante_ms=50):
        """
        Corrige el delay entre los picos detectados en la señal integrada (convolv)
        y su ubicación real en la señal filtrada (seg), buscando el máximo real
        dentro de una ventana alrededor de cada candidato.
        """
        muestras_atras = int(ventana_atras_ms / 1000 * record_fs)
        muestras_adelante = int(ventana_adelante_ms / 1000 * record_fs)

        picos_refinados = []

        for i, p in enumerate(picos_candidatos):
            # --- límite dinámico hacia atrás ---
            if i == 0:
                # primer pico: no hay vecino anterior, solo respeta el borde del arreglo
                limite_atras = max(0, p - muestras_atras)
            else:
                punto_medio_anterior = (picos_candidatos[i - 1] + p) // 2
                limite_atras = max(punto_medio_anterior, p - muestras_atras)

            # --- límite dinámico hacia adelante ---
            if i == len(picos_candidatos) - 1:
                # último pico: no hay vecino siguiente, solo respeta el borde del arreglo
                limite_adelante = min(len(seg), p + muestras_adelante)
            else:
                punto_medio_siguiente = (p + picos_candidatos[i + 1]) // 2
                limite_adelante = min(punto_medio_siguiente, p + muestras_adelante)

            segmento = seg[limite_atras:limite_adelante]

            # buscamos el máximo en valor absoluto dentro de ESE segmento acotado
            indice_local = np.argmax(np.abs(segmento))
            indice_global = limite_atras + indice_local

            picos_refinados.append(indice_global)
        # print(np.array(picos_refinados))
        return np.array(picos_refinados)


    def comprobar(self, picos, time, annotation, seg, record, graficar=False, duracion = None):
        #Ahora, podemos comprobar con las anotaciones del registro, para esto, podemos usar la función rdann de wfdb, y el archivo de anotaciones correspondiente al registro.
        if duracion == None:
            n_muestras = len(seg)
            picos_reales = annotation.sample[annotation.sample < n_muestras]
        else:
            picos_reales = annotation.sample[annotation.sample < duracion * record.fs]
        tiempos_detectados = time[:-1][picos]
        tiempos_reales = picos_reales / record.fs
        if graficar == True:
            plt.plot(time, seg)
            plt.scatter(tiempos_reales, [seg.max()] * len(tiempos_reales), marker='x', color='r')
            plt.scatter(tiempos_detectados, [seg.max() * 1.1] * len(tiempos_detectados), marker='o', color='g')
            plt.legend(['Señal de ECG', 'Picos R detectados', 'Picos R reales'])
            plt.title("Reporte final")
            plt.show()


if  __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    dir_mitdb = PROJECT_ROOT / "data" / "mitdb" /"100"
    ecg = data(dir_mitdb)
    pan_tompkins = Pan_tompkins(ecg.record_path)
    picos, time, seg, convolv = pan_tompkins.find_peaks(ecg.record, graficar=True, duracion=10)
    pan_tompkins.comprobar(picos, time, ecg.annotation, seg, ecg.record, True)