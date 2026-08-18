import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append('.')
from data_loader import Loader
from find_peaks_pan_tompkins import Pan_tompkins
from beat_segmenter import Beat_segmenter

PROJECT_ROOT = Path(__file__).parent.parent
dir_mitdb = PROJECT_ROOT / "data" / "mitdb"

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
    df_registro['registro'] = r
    
    tablas.append(df_registro)

df_completo = pd.concat(tablas, ignore_index=True)
print(df_completo['Etiqueta'].value_counts())

# 1. Quitar filas sin etiqueta confiable (los None que vimos antes)
df_limpio = df_completo.dropna(subset=['Etiqueta'])

print(f"Filas originales: {len(df_completo)}")
print(f"Filas después de limpiar: {len(df_limpio)}")
df_limpio['Etiqueta_binaria'] = df_limpio['Etiqueta'].apply(lambda e: 'N' if e == 'N' else 'Anomalo')
print(df_limpio['Etiqueta_binaria'].value_counts())

# 2. Separar features (X) de la etiqueta a predecir (y)
X = df_limpio[['Amplitud', 'Intervalo_RR']]
y = df_limpio['Etiqueta_binaria']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X.head())
print(y.value_counts())
print("Train:", y_train.value_counts())
print("Test:", y_test.value_counts())

# Crear y entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba (datos que el modelo NUNCA vio en entrenamiento)
y_pred = modelo.predict(X_test)

# Evaluar
print("=== Matriz de confusión ===")
print(confusion_matrix(y_test, y_pred, labels=['N', 'Anomalo']))

print("\n=== Reporte de clasificación ===")
print(classification_report(y_test, y_pred))