# Predicción del Tipo de Metástasis en Pacientes con Cáncer de Mama Triple Negativo (TNBC) mediante Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![WEKA](https://img.shields.io/badge/WEKA-000000?style=for-the-badge)

Proyecto de Minería de Datos — **Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana (2026)**

---

## Prueba el modelo desplegado

> ### [**despliegue-cancermama-gpjevuer6gnc8jldzcbwyw.streamlit.app**](https://despliegue-cancermama-gpjevuer6gnc8jldzcbwyw.streamlit.app)
>
> Ingresa el ZIP3, la edad, el tipo de pagador y el código de diagnóstico de una paciente, y la aplicación predice el **tipo de metástasis más probable** en tiempo real.

[![Abrir la App](https://img.shields.io/badge/Abrir_la_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://despliegue-cancermama-gpjevuer6gnc8jldzcbwyw.streamlit.app)

---

## Descripción del proyecto

El **cáncer de mama triple negativo (TNBC)** es un subtipo de cáncer de mama invasivo que representa entre el 10 % y el 15 % de todos los casos de carcinoma mamario. Se caracteriza por la ausencia de los marcadores ER, PR y HER2, lo que limita las opciones de tratamiento y le da un comportamiento clínico más agresivo, con tasas de supervivencia inferiores y mayor riesgo de recurrencia temprana.

Cuando el cáncer se disemina (metástasis), la localización del nuevo tumor condiciona directamente el pronóstico y el tratamiento. Identificar tempranamente el **tipo de metástasis probable** puede aportar al manejo clínico de las pacientes.

Este proyecto busca responder la pregunta:

> **¿Es posible predecir el tipo de metástasis que desarrollará una paciente con TNBC a partir de características clínicas, demográficas, socioeconómicas y ambientales?**

## Objetivos

**Objetivo general:** evaluar si a partir de variables clínicas, demográficas, socioeconómicas y ambientales se pueden desarrollar modelos analíticos que permitan identificar perfiles y predecir el tipo de metástasis en pacientes con TNBC.

**Objetivos específicos:**
- Analizar qué características tienen mayor relación con el tipo de metástasis.
- Evaluar distintos modelos predictivos para clasificar el tipo de metástasis.
- Perfilar pacientes con TNBC en búsqueda de patrones mediante clustering.

## Datos

| | |
|---|---|
| **Fuente** | WiDS Datathon 2024 (Kaggle), datos suministrados por Gilead Sciences y Health Verity |
| **Enriquecimiento** | Datos geo-demográficos (terceros) y datos de calidad del aire a nivel de ZIP (NASA / Universidad de Columbia) |
| **Cobertura** | Pacientes reales anonimizadas, diagnosticadas con TNBC en EE. UU. entre 2015 y 2018 |
| **Tamaño inicial** | 9.412 observaciones × 83 variables (de un total de 18.698 registros combinando entrenamiento y prueba de WiDS Datathon 2024, tras filtrar por registros con `patient_race` no nulo) |
| **Variable objetivo** | `metastatic_cancer_diagnosis_code` (ICD-10-CM), recodificada en `tipo_de_metastasis` |

La variable objetivo se agrupó en **3 categorías** según el sistema orgánico comprometido:

| Sistema orgánico | Códigos ICD-10 | Ejemplos |
|---|---|---|
| Sistema Linfático | C77.0 – C77.9 | Ganglios axilares, ganglios no especificados |
| Sistema Esquelético | C79.51 | Metástasis ósea |
| Visceral / Sistémico | C78.0 – C78.9, C79.x (excepto C79.51) | Metástasis pulmonar, hepática, mamaria |

La clase mayoritaria (sistema linfático) representa el **63,70 %** de los casos — este valor se usó como línea base de referencia para evaluar los modelos.

### Archivos de datos en el repositorio (carpeta `datos/`)

- **`datos_maestros.csv`**: dataset base con los 9.412 registros y todas las variables restantes tras descartar inicialmente las irrelevantes (`patient_id`, `region`, `zip3`, entre otras). Se utilizó para el análisis de factores (selección de variables) y como base para el balanceo con SMOTE.
- **Archivos `limpieza_weka`**: resultado de la imputación de nulos realizada en WEKA sobre `datos_maestros.csv`. A partir de estos datos se realizó el modelo descriptivo (clustering).

`datos_maestros.csv` y los archivos `limpieza_weka` son prácticamente iguales; la diferencia es que `datos_maestros.csv` conservaba algunas columnas adicionales que posteriormente también fueron eliminadas durante la selección de variables.

## Preparación de los datos

Todo el proceso de limpieza inicial se realizó en **WEKA**:

- **Eliminadas:** `patient_id`, `patient_gender` (todas las pacientes son mujeres), variables geográficas redundantes (se conservó solo `Region`), variables con >15 % de nulos (`bmi`), y variables con **data leakage** (`DiagPeriodL90D`, `metastatic_first_novel_treatment*`).
- **Imputadas:** variables con hasta 15 % de valores faltantes (ingresos, vivienda, educación, estado civil, entre otras).
- **Selección de variables:** se eliminaron variables con multicolinealidad (correlación > 0.8 o < −0.8) y variables con relación lineal/no lineal menor a 0.002 respecto al objetivo. El dataset final quedó en **73 variables predictoras**.
- **Balanceo:** SMOTE sobre la clase minoritaria (esquelético), generando 276 registros sintéticos (de 1.383 a 1.659 registros).
- **Ingeniería de características:** la variable `breast_cancer_diagnosis_code` (50 clases) se agrupó en 6 categorías según localización anatómica del tumor primario.
- **Transformaciones finales:** `LabelEncoder` para el objetivo, `StandardScaler` para variables numéricas, `One-Hot Encoding` para categóricas, integradas con `ColumnTransformer`. Partición estratificada 70/30 y validación cruzada para la evaluación final.

## Modelado y resultados

### Clasificación

Se probaron 4 modelos en una partición 70/30:

| Método | Precision | Recall | F1-Score |
|---|---|---|---|
| Red Neuronal | 0.56 | 0.33 | 0.25 |
| K-NN | 0.41 | 0.40 | 0.40 |
| SVM | 0.41 | 0.41 | 0.39 |
| **Random Forest** | **0.43** | **0.44** | **0.42** |

Random Forest y K-NN se profundizaron con validación cruzada y distintas configuraciones:

| Algoritmo | Configuración | Precision | Recall | F1-Score | Fit-time (s) |
|---|---|---|---|---|---|
| Random Forest | `n_estimators=200, max_depth=10` | 0.43 | 0.44 | 0.42 | 4.42 |
| Random Forest | `n_estimators=400, max_depth=15` | 0.43 | 0.44 | 0.43 | 10.75 |
| K-NN | `k=4` | 0.41 | 0.39 | 0.39 | 0.03 |
| K-NN | `k=10` | 0.40 | 0.37 | 0.36 | 0.03 |

> **Nota honesta:** ninguno de los modelos logró un desempeño alto, principalmente por la débil relación entre las variables disponibles y la variable objetivo, sumado al desbalance entre clases. Aun así, se seleccionó **Random Forest (`n_estimators=200`, `max_depth=10`)** para el despliegue por ofrecer el mejor balance entre métricas y costo computacional de entrenamiento.

### Clustering

Se aplicó **K-Means** (k=6, seleccionado mediante método del codo y de la rodilla) sobre variables clínicas, sociodemográficas y ambientales (89 variables tras codificación one-hot). Inercia: 30.506 · Coeficiente de silueta: 0.106 (bajo, por lo que los perfiles deben interpretarse como tendencias generales y no como segmentos claramente diferenciados).

| Clúster | Perfil | Características principales |
|---|---|---|
| 0 | Zonas suburbanas | Edad promedio 54 años, población moderada, estructura social estable |
| 1 | Entornos urbanos | Edad promedio 61 años, alta densidad, 38 % nunca casada |
| 2 | Entorno rural tradicional | Edad promedio 63 años, baja densidad, 51 % casada |
| 3 | Zonas urbanas solteras | Edad promedio 58 años, 40 % nunca casada |
| 4 | Zonas urbanas de perfil promedio | Edad promedio 59 años, alta población |
| 5 | Comunidades semi-rurales y periféricas | Edad promedio 59 años, mayor proporción divorciada/viuda |

## Despliegue

El modelo Random Forest y el `LabelEncoder` entrenados se serializaron en formato `.pkl` y se integraron en una aplicación web con **Streamlit**.

La app enriquece automáticamente los datos: el usuario solo ingresa **ZIP3, edad, tipo de pagador y código de diagnóstico**, y la aplicación busca en `zip_completo.csv` las variables socioeconómicas asociadas a ese ZIP3, arma el dataframe de 39 variables que espera el modelo, genera la predicción y decodifica el resultado a su categoría original.

### Pruébala aquí: [despliegue-cancermama-gpjevuer6gnc8jldzcbwyw.streamlit.app](https://despliegue-cancermama-gpjevuer6gnc8jldzcbwyw.streamlit.app)

## Estructura del repositorio

```
├── datos/                          # datos_maestros.csv y archivos limpieza_weka (ver sección Datos)
├── despliegue_cancer_mama-main/    # Código fuente de la app de Streamlit
├── documentos/                     # Documento técnico completo del proyecto
├── modelos/                        # Modelos entrenados (.pkl) y encoders
├── seleccion_factores.ipynb        # Selección de variables (correlación / multicolinealidad)
├── smoteador.ipynb                 # Balanceo de clases con SMOTE
├── probando_modelos_70_30.ipynb    # Comparación inicial de modelos (70/30)
├── modelado_CV.ipynb               # Modelado con validación cruzada
├── modelo_final.ipynb              # Entrenamiento y exportación del modelo final
├── Clustering_REVISION.ipynb       # Segmentación de pacientes con K-Means
└── requirements.txt                # Dependencias del proyecto
```

## Stack tecnológico

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `imbalanced-learn (SMOTE)` · `WEKA` (preparación de datos) · `Streamlit` (despliegue) · `Pickle` (serialización de modelos)

## Autores

Proyecto desarrollado para el curso de Minería de Datos — Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana (2026).

- **Jhonier Elibardo Correa Ríos** — [GitHub: JOTAKR8](https://github.com/JOTAKR8)
- **Jose Daniel Arenas Quintero**
- **Ines Margarita González Pontón**
- **Docente:** Ana Isabel Oviedo Carrascal

## Consideraciones y limitaciones

- Este es un proyecto **académico**, desarrollado con fines educativos; no debe usarse como herramienta de diagnóstico ni apoyo a decisiones clínicas reales.
- Los modelos de clasificación no superaron ampliamente la línea base de la clase mayoritaria, lo que refleja la complejidad del problema y la necesidad de variables con mayor poder predictivo.
- El coeficiente de silueta bajo en el clustering (0.106) indica que los perfiles encontrados son tendencias generales de la población, no segmentos claramente separables.
- La variable de raza se excluyó del modelo tras el análisis de correlación, evitando su uso como predictor.
