# ECG Anomaly Detector

> Pipeline completo de procesamiento de señales ECG: filtrado, detección de picos R (Pan-Tompkins), extracción de features y clasificación de latidos anómalos, usando el MIT-BIH Arrhythmia Database.


## Motivación

Como estudiante de Ingeniería Biomédica/Biónica, quería construir un proyecto que combinara procesamiento de señales, ingeniería de features clínicamente relevantes, y Machine Learning aplicado a datos reales — no un dataset genérico de práctica, sino señales fisiológicas reales anotadas por cardiólogos.

## Qué hace

1. **Carga** registros de ECG anotados desde PhysioNet (MIT-BIH Arrhythmia Database).
2. **Filtra** la señal cruda con un filtro pasa-banda Butterworth (0.5–40 Hz), eliminando ruido de línea eléctrica y deriva de línea base.
3. **Detecta picos R** con una implementación propia y simplificada del algoritmo Pan-Tompkins (derivada → cuadrado → integración por ventana móvil → umbral adaptativo).
4. **Segmenta cada latido** y extrae features: amplitud normalizada por paciente, intervalo RR, y ancho del complejo QRS.
5. **Clasifica** cada latido como Normal o Anómalo con un Random Forest, evaluado con validación cruzada agrupada por paciente (`GroupKFold`) para evitar fuga de datos entre pacientes.

## Arquitectura del pipeline

```
Loader → Filter → PanTompkins → BeatSegmenter → RandomForestClassifier
```

Cada etapa es una clase independiente con una única responsabilidad, permitiendo probar y depurar cada paso por separado antes de integrarlo al pipeline completo.

## Dataset

- **Fuente:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), PhysioNet.
- **Registros usados:** 30 registros completos (~30 min c/u), cubriendo pacientes con ritmo normal y con arritmias diversas.
- **Total de latidos procesados:** 53,066 (después de descartar latidos sin correspondencia confiable con las anotaciones del cardiólogo).
- **Distribución:** ~75% latidos normales (N), ~25% anómalos (mezcla de bloqueos de rama, contracciones prematuras, latidos de marcapasos, y otros tipos).

## Resultados

Evaluación con `GroupKFold` (5 folds, agrupado por paciente — ningún latido del mismo paciente aparece en train y test simultáneamente, evitando fuga de datos):

- **Recall promedio (macro):** 0.75
- **Feature más predictivo:** Ancho del QRS (42% de importancia), seguido de Intervalo RR (32%) y Amplitud normalizada (26%)

**Tasa de detección por tipo de anomalía:**

| Tipo de latido | Descripción | Tasa de detección |
|---|---|---|
| V | Contracción ventricular prematura | 96.7% |
| / | Latido de marcapasos | 96.7% |
| Q | Latido no clasificable | 75.0% |
| f | Fusión latido normal/marcapasos | 70.6% |
| L | Bloqueo de rama izquierda | 31.7% |
| R | Bloqueo de rama derecha | 23.1% |

El modelo detecta con alta fiabilidad anomalías que alteran claramente el ritmo o la amplitud (contracciones prematuras, marcapasos). Las categorías con menor detección (bloqueos de rama) tienen una morfología de QRS multifásica que el método actual de medición de ancho (umbral de media amplitud) subestima — ver Limitaciones.

## Limitaciones 

- **Bloqueos de rama (R, L) siguen siendo difíciles de detectar.** Su morfología de QRS multifásica (múltiples deflexiones) hace que un umbral simple de media amplitud subestime el ancho real del complejo. Una mejora futura sería medir el ancho sobre la señal ya integrada de Pan-Tompkins (el "envelope" de energía), que captura mejor complejos multifásicos que la señal cruda.
- **Clasificación binaria (Normal/Anómalo), no multi-clase.** Se simplificó así por el tamaño limitado de algunas categorías (algunas con 1-2 ejemplos totales en el dataset).
- **Solo 2 derivaciones de ECG disponibles en MIT-BIH**, se usó únicamente el canal 0; una señal multi-derivación real podría mejorar la detección de morfologías complejas.
- **No es una herramienta de diagnóstico clínico.** Es un proyecto educativo/de portafolio; cualquier uso clínico real requeriría validación regulatoria extensiva.

## Cómo correrlo

```bash
git clone <tu-repo>
cd ecg-anomaly-detector
pip install -r requirements.txt

python src/data_loader.py      # descarga los registros del MIT-BIH necesarios, en la clase Loader ya viene una sección que los descargará si no los tienes aún
python src/clasifier.py        # corre el pipeline completo y entrena el modelo
```

## Stack técnico

- `wfdb` — lectura de señales y anotaciones de PhysioNet
- `numpy` / `scipy` — procesamiento de señal (filtrado, convolución)
- `pandas` — estructuración de features por latido
- `scikit-learn` — clasificación (Random Forest) y validación cruzada agrupada

## Próximos pasos

- [ ] Medir ancho de QRS sobre la señal integrada (mejora esperada en R/L)
- [ ] Comparar Random Forest contra XGBoost / Gradient Boosting
- [ ] Dashboard interactivo con Streamlit
- [ ] Clasificación multi-clase (más allá de binaria)

## Referencias

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol* 20(3):45-50, 2001.
- Pan J, Tompkins WJ. A Real-Time QRS Detection Algorithm. *IEEE Trans. Biomed. Eng.* BME-32(3), 1985.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), 2000.
