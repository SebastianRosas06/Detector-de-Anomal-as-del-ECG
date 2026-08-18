import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


#wfdb.dl_database('mitdb', dl_dir='../src/data/mitdb', records=['100', '106', '200'])
PROJECT_ROOT = Path(__file__).parent.parent
class Loader:

    def __init__(self, record_path):
        self.record_path = str(record_path)
        self.record, self.annotation = self.analisis()
    
    def analisis(self):
        print("Obteniendo record")
        record = wfdb.rdrecord(self.record_path)
        print("Obteniendo annotation")
        annotation = wfdb.rdann(self.record_path, 'atr')
        print("Record:",record," Annotation:",annotation)
        return record, annotation

    def graficar(self, segundos):
        fin = int(segundos * self.record.fs)
        seg = self.record.p_signal[0:fin, 0]
        time = np.arange(0, fin) / self.record.fs
        plt.plot(time, seg)
        picos = self.annotation.sample[self.annotation.sample < fin]
        timehigh = picos / self.record.fs
        plt.scatter(timehigh, seg[picos], color='red', label='R-peaks')
        plt.title('ECG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def graficarRange(self, inicio, fin):
        seg = self.record.p_signal[inicio:fin, 0]
        time = np.arange(inicio, fin) / self.record.fs
        plt.plot(time, seg)
        picos = self.annotation.sample[(self.annotation.sample < fin) & (self.annotation.sample > inicio)] - inicio
        timehigh = picos / self.record.fs
        plt.scatter(timehigh, seg[picos], color='red', label='R-peaks')
        plt.title('ECG Signal with limits')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    dir_mitdb = PROJECT_ROOT / "data" / "mitdb"
    
    registros_necesarios = ["100", "101", "103", "105", "112", "113", "115", "121", "220", "230",
    "106", "116", "119", "200", "203", "205", "213", "215", "233",
    "201", "202", "210", "217", "221",
    "109", "111", "118", "124", "212",
    "108"]
    
    faltantes = [r for r in registros_necesarios if not (dir_mitdb / f"{r}.hea").exists()]
    
    if faltantes:
        print(f"Descargando registros faltantes: {faltantes}")
        os.makedirs(dir_mitdb, exist_ok=True)
        wfdb.dl_database("mitdb", dl_dir=str(dir_mitdb), records=faltantes)
        print("¡Descarga completada!")
    else:
        print("Todos los registros ya están descargados.")

    # 1. Crear la instancia cargando los datos
    ecg = Loader(dir_mitdb / "100")

    # 2. Graficar los primeros 5 segundos
    ecg.graficar(5)

    # 3. Graficar un rango personalizado por muestras
    ecg.graficarRange(inicio=1000, fin=3000)
