import wfdb
import numpy as np
import matplotlib.pyplot as plt

def analisis(path):
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')
    return record, annotation

def graficar(record, annotation):
    fin = int(5 * record.fs)
    seg = record.p_signal[0:fin, 0]
    time = np.arange(0, fin) / record.fs
    plt.plot(time, seg)
    picos = annotation.sample[annotation.sample < fin]
    timehigh = picos / record.fs
    plt.scatter(timehigh, seg[picos], color='red', label='R-peaks')
    plt.title('ECG Signal with R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


record, annotation = analisis('data/mitdb/100')