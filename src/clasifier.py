import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.model_selection import GroupKFold, cross_val_predict
sys.path.append('.')
from data_loader import Loader
from find_peaks_pan_tompkins import Pan_tompkins
from beat_segmenter import Beat_segmenter
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
dir_mitdb = PROJECT_ROOT / "data" / "mitdb"

#Carga de registros
registros = ["100", "101", "103", "105", "112", "113", "115", "121", "220", "230",
    "106", "116", "119", "200", "203", "205", "213", "215", "233",
    "201", "202", "210", "217", "221",
    "109", "111", "118", "124", "212",
    "108"]
tablas = []

for r in registros:
    path_registro = dir_mitdb / r
    ecg = Loader(str(path_registro))  # ruta absoluta, no depende del cwd
    pt = Pan_tompkins(ecg.record_path)
    picos, time, seg, convolv = pt.find_peaks(ecg.record)
    
    segmentador = Beat_segmenter(seg, picos, ecg.annotation, ecg.record.fs)
    df_registro = segmentador.tabular(picos, ecg.annotation, ecg.record.fs)
    #Añadimos al vector en la ubicación del registro, todo lo que hicimos operando a dicho registro
    df_registro['registro'] = r
    
    tablas.append(df_registro)

df_completo = pd.concat(tablas, ignore_index=True)
print(df_completo['Etiqueta'].value_counts())

#Quitamos las filas sin etiqueta confiable (None)
df_limpio = df_completo.dropna(subset=['Etiqueta'])

print(f"Filas originales: {len(df_completo)}")
print(f"Filas después de limpiar: {len(df_limpio)}")
#Aquí determinamos que sólo necesitamos saber si el latido es normal o anómalo
df_limpio['Etiqueta_binaria'] = df_limpio['Etiqueta'].apply(lambda e: 'N' if e == 'N' else 'Anomalo')
print(df_limpio['Etiqueta_binaria'].value_counts())
df_limpio['Amplitud_norm'] = df_limpio.groupby('registro')['Amplitud'].transform(
    lambda x: (x - x.mean()) / x.std()
)


#Separamos los features (X) de la etiqueta a predecir (y)
X = df_limpio[['Amplitud_norm', 'Intervalo_RR', 'Ancho_QRS']]
y = df_limpio['Etiqueta_binaria']
grupos = df_limpio['registro']
#Uso de GroupKFold para una mejor evaluación, reemplazando train_test_split
gkf = GroupKFold(n_splits=5)
modelo_cv = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
scores = cross_val_score(modelo_cv, X, y, groups=grupos, cv=gkf, scoring='recall_macro')
print("Recall por fold (agrupado por registro):", scores)
print("Promedio:", scores.mean())

# Entrena un modelo final con los datos normalizados para ver feature importance real
modelo_final = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo_final.fit(X, y)  # aquí puedes entrenar con todo, ya que es solo para inspeccionar importancia
importancias = pd.Series(modelo_final.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Importancia de features (modelo final) ===")
print(importancias)

y_pred_cv = cross_val_predict(modelo_cv, X, y, groups=grupos, cv=gkf)

# Compara contra la etiqueta ORIGINAL (multi-clase), no la binaria,
# para ver qué tipos específicos se confunden más
df_analisis = df_limpio.copy()
df_analisis['prediccion'] = y_pred_cv

# Para cada tipo original de latido, ¿qué tan seguido el modelo acertó (predijo Anomalo)?
tasa_deteccion = df_analisis[df_analisis['Etiqueta_binaria'] == 'Anomalo'].groupby('Etiqueta').apply(
    lambda grupo: (grupo['prediccion'] == 'Anomalo').mean()
)
print("Tasa de detección por tipo específico de anomalía:")
print(tasa_deteccion.sort_values())
print(df_limpio.groupby('Etiqueta')['Ancho_QRS'].describe())

# Encuentra un pico etiquetado como 'R' para inspeccionar
idx_R = df_limpio[df_limpio['Etiqueta'] == 'R'].index[0]
pico_R = df_limpio.loc[idx_R, 'Pico']

latido = segmentador.segmentar_latido(pico_R)
indice_pico = np.argmax(np.abs(latido))
plt.plot(latido)
plt.axhline(y=latido[indice_pico]/2, color='gray', linestyle='--', label='umbral')
plt.axvline(x=indice_pico, color='green', linestyle=':', label='pico detectado')
plt.title(f"Latido R en pico {pico_R}, ancho: {segmentador.calcular_qrs(pico_R):.1f}ms")
plt.legend()
plt.show()