import numpy as np;
import matplotlib.pyplot as plt
from data_loader import Loader as data;
import pandas as pd;
class Beat_segmenter():

    def __init__(self, signal, picos, annotation, fs, ventana_sec=0.2, tolerancia_ms=100):
        self.signal = signal #Señal filtrada
        self.picos = picos #Picos R detectados
        self.annotation = annotation #Anotaciones del registro
        self.fs = fs #Frecuencia de muestreo
        self.ventana_muestras = int(ventana_sec * fs)  # Indica la mitad de la duración de la ventana por latido
        self.tolerancia_muestras = int(tolerancia_ms * fs / 1000)  # Convertir tolerancia a muestras

    def segmentar_latido(self, pico_prueba):
        inicio = max(0, pico_prueba - self.ventana_muestras)
        fin = min(len(self.signal), pico_prueba + self.ventana_muestras)    
        plt.plot([inicio,fin], [self.signal[inicio], self.signal[fin]], color='red', label='Latido segmentado')
        return self.signal[inicio:fin]

    def calcular_amplitudes(self):
        amplitudes = []
        for pico in self.picos:
            inicio = max(0, pico - self.ventana_muestras)
            fin = min(len(self.signal), pico + self.ventana_muestras)
            amplitudes.append(np.max(self.signal[inicio:fin]))
        return amplitudes
    @staticmethod
    def calcular_intervalos_rr(picos, fs):
        if len(picos) < 2:
            return np.zeros(len(picos))  # no hay suficientes picos para calcular RR
        
        rr = np.zeros(len(picos))
        for i in range(1, len(picos)):
            rr[i] = (picos[i] - picos[i - 1]) / fs
        rr[0] = rr[1]
        return rr
    
    @staticmethod
    def emparejar_etiquetas(picos, annotation, fs, tolerancia_ms=100):
        tolerancia_muestras = int(tolerancia_ms * fs / 1000)
        etiquetas = []
        for pico in picos:
            # Buscar anotaciones dentro de la ventana de tolerancia
            diffs = np.abs(annotation.sample - pico)
            if np.any(diffs <= tolerancia_muestras):
                etiqueta = annotation.symbol[np.argmin(diffs)]
            else:
                etiqueta = None  # No hay anotación cercana, etiquetamos como normal
            etiquetas.append(etiqueta)
        return etiquetas
    
    def tabular(self, picos, annotation, fs):
        amplitudes = self.calcular_amplitudes()
        rr_intervals = self.calcular_intervalos_rr(picos, fs)
        etiquetas = self.emparejar_etiquetas(picos, annotation, fs)
        tablas = []
        for i in range(len(picos)):
            tablas.append({
                'Pico': picos[i],
                'Amplitud': amplitudes[i],
                'Intervalo_RR': rr_intervals[i],
                'Etiqueta': etiquetas[i]
            })
        df = pd.DataFrame(tablas)  
        return df


if __name__ == "__main__":
    import sys
    sys.path.append('.')  # ajusta según tu estructura real
    from data_loader import Loader
    from find_peaks_pan_tompkins import Pan_tompkins

    ecg = Loader('../data/mitdb/100')
    pt = Pan_tompkins(ecg.record_path)
    picos, time, seg, convolv = pt.find_peaks(ecg.record)

    segmentador = Beat_segmenter(seg, picos, ecg.annotation, ecg.record.fs)
    tabla = segmentador.tabular(picos, ecg.annotation, ecg.record.fs)

    for indice, fila in tabla.iterrows():
        print(f"Latido en índice {indice}: amplitud={fila['Amplitud']}, etiqueta={fila['Etiqueta']}")