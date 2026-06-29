# 📊 Tarea: Investigación y Desarrollo de Algoritmos de Clasificación en Machine Learning Supervisado

Deben investigar cómo funcionan los algoritmos de clasificación en Machine Learning supervisado.

---

## Parte 1: Algoritmos de Clasificación en ML Supervisado

---

### 1. ¿Qué es la clasificación en Machine Learning y cuál es su propósito? Añade un ejemplo práctico de un problema de clasificación del mundo real donde sea necesario aplicar estos algoritmos.

La **clasificación** en Machine Learning es una tarea de aprendizaje supervisado cuyo objetivo es asignar una **etiqueta o categoría** a una entrada de datos, basándose en patrones aprendidos a partir de ejemplos etiquetados previamente.

El propósito principal es entrenar un modelo capaz de aprender la relación entre las características (features) de los datos y sus categorías correspondientes, para luego predecir la categoría correcta de nuevas instancias no vistas.

**Ejemplo práctico — Detección de correos spam:**

En los servicios de correo electrónico (Gmail, Outlook, etc.), se aplican algoritmos de clasificación para determinar si un correo entrante es **spam** o **no spam**. El modelo analiza características como:

- Palabras clave en el asunto y cuerpo del mensaje
- Dirección del remitente
- Frecuencia de enlaces externos
- Presencia de archivos adjuntos sospechosos

Con estos datos, el modelo aprende patrones y clasifica automáticamente cada nuevo correo, protegiendo al usuario sin intervención manual.

---

### 2. ¿Cuál es la diferencia entre clasificación binaria y multiclase? Proporciona ejemplos de cada tipo y explica cuándo usar cada uno.

| Característica | Clasificación Binaria | Clasificación Multiclase |
|---|---|---|
| Número de categorías | 2 | 3 o más |
| Salida del modelo | 0 o 1 / Sí o No | Una categoría de entre N posibles |
| Complejidad | Menor | Mayor |

**Clasificación Binaria** — el modelo predice una de **dos posibles clases**:

- Diagnóstico médico: ¿el tumor es maligno o benigno?
- Aprobación de crédito: ¿el cliente pagará o no pagará el préstamo?
- Detección de fraude: ¿la transacción es fraudulenta o legítima?

*Cuándo usarla:* cuando el problema tiene exactamente dos resultados posibles y excluyentes entre sí.

**Clasificación Multiclase** — el modelo predice una de **tres o más clases**:

- Reconocimiento de dígitos escritos a mano: clasificar entre los dígitos del 0 al 9
- Clasificación de noticias: deportes, política, tecnología, cultura, economía
- Diagnóstico de enfermedades: COVID-19, gripe, resfriado común, neumonía bacteriana

*Cuándo usarla:* cuando el problema tiene más de dos categorías posibles y cada instancia pertenece exactamente a una de ellas.

---

### 3. ¿Cuáles son los principales algoritmos de clasificación supervisada?

#### Regresión Logística

A pesar de su nombre, es un algoritmo de **clasificación**. Modela la probabilidad de que una instancia pertenezca a una clase utilizando la función sigmoide, que transforma cualquier valor numérico en un rango entre 0 y 1.

- **Cómo funciona:** calcula una combinación lineal de las features y aplica la función sigmoide para obtener una probabilidad. Si la probabilidad supera un umbral (generalmente 0.5), se asigna la clase positiva.
- **Ventajas:** simple, interpretable, eficiente con datos linealmente separables.
- **Limitaciones:** no funciona bien con relaciones no lineales entre variables.
- **Uso típico:** clasificación de spam, predicción de abandono de clientes (churn).

#### Árboles de Decisión

Son modelos que dividen el espacio de los datos mediante una serie de **preguntas jerárquicas** (nodos), siguiendo una estructura de árbol hasta llegar a una predicción (hoja).

- **Cómo funciona:** el algoritmo selecciona en cada nodo la feature y el umbral que mejor separa las clases, utilizando criterios como la **Ganancia de Información** o el **índice Gini**.
- **Ventajas:** altamente interpretables, no requieren normalización de datos, manejan variables categóricas y numéricas.
- **Limitaciones:** propensos al sobreajuste (overfitting) si no se controla su profundidad.
- **Uso típico:** análisis de riesgo crediticio, diagnóstico médico.

#### Support Vector Machine (SVM)

SVM busca el **hiperplano óptimo** que maximiza el margen entre las clases en el espacio de características. Los puntos más cercanos al hiperplano se denominan **vectores de soporte**.

- **Cómo funciona:** proyecta los datos (mediante funciones kernel como RBF o polinomial) a un espacio de mayor dimensión donde sean linealmente separables, y encuentra el hiperplano con mayor margen posible.
- **Ventajas:** muy efectivo en espacios de alta dimensión, robusto frente a outliers.
- **Limitaciones:** costoso computacionalmente en conjuntos de datos muy grandes, difícil de interpretar.
- **Uso típico:** clasificación de imágenes, reconocimiento de texto, bioinformática.

#### K-Nearest Neighbors (KNN)

KNN es un algoritmo **basado en instancias** que clasifica un nuevo punto según las clases de sus *K* vecinos más cercanos en el espacio de características.

- **Cómo funciona:** dado un nuevo punto, calcula la distancia (generalmente Euclidiana) a todos los puntos del conjunto de entrenamiento, selecciona los K más cercanos y asigna la clase mayoritaria entre ellos.
- **Ventajas:** simple de entender e implementar, no requiere entrenamiento explícito.
- **Limitaciones:** lento en predicción con datasets grandes, sensible a la escala de las variables y a datos irrelevantes.
- **Uso típico:** sistemas de recomendación, reconocimiento de patrones.

#### Naive Bayes

Basado en el **Teorema de Bayes**, asume independencia condicional entre las features dado el valor de la clase (de ahí el término "naive" o ingenuo).

- **Cómo funciona:** calcula la probabilidad posterior de cada clase dado un conjunto de features usando el teorema de Bayes, y asigna la clase con mayor probabilidad.
- **Ventajas:** extremadamente rápido, funciona bien con pocos datos, muy efectivo en texto.
- **Limitaciones:** la asunción de independencia rara vez se cumple en la realidad, lo que puede afectar la precisión.
- **Uso típico:** filtrado de spam, análisis de sentimientos, clasificación de documentos.

---

### 4. ¿Qué métricas se utilizan para evaluar modelos de clasificación?

#### Accuracy (Exactitud)

Proporción de predicciones correctas sobre el total de predicciones realizadas.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- **Cuándo usarla:** cuando las clases están **balanceadas**.
- **Limitación:** puede ser engañosa en datasets desbalanceados. Un modelo que prediga siempre la clase mayoritaria puede tener alta accuracy sin ser útil.

#### Precision (Precisión)

De todos los casos que el modelo predijo como positivos, ¿cuántos realmente lo eran?

```
Precision = TP / (TP + FP)
```

- **Cuándo es importante:** cuando el costo de los **falsos positivos** es alto. Por ejemplo, en detección de spam (no queremos marcar como spam correos legítimos).

#### Recall (Sensibilidad)

De todos los casos que realmente eran positivos, ¿cuántos detectó el modelo correctamente?

```
Recall = TP / (TP + FN)
```

- **Cuándo es importante:** cuando el costo de los **falsos negativos** es alto. Por ejemplo, en diagnóstico de enfermedades graves (no queremos no detectar un caso real).

#### F1-Score

Media armónica entre Precision y Recall. Equilibra ambas métricas en un solo valor.

```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Cuándo usarla:** cuando se necesita un balance entre Precision y Recall, especialmente en datasets desbalanceados.
- Valores próximos a 1 indican un modelo de alta calidad.

#### Matriz de Confusión

Tabla que resume el rendimiento del modelo mostrando las predicciones correctas e incorrectas por clase:

|  | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | TP (Verdadero Positivo) | FN (Falso Negativo) |
| **Real Negativo** | FP (Falso Positivo) | TN (Verdadero Negativo) |

- **TP:** el modelo predijo positivo y era positivo.
- **TN:** el modelo predijo negativo y era negativo.
- **FP:** el modelo predijo positivo pero era negativo (error tipo I).
- **FN:** el modelo predijo negativo pero era positivo (error tipo II).

Permite visualizar de forma clara dónde falla el modelo y en qué tipo de errores incurre con mayor frecuencia.

#### ROC-AUC

- **Curva ROC (Receiver Operating Characteristic):** gráfico que muestra la tasa de verdaderos positivos (Recall) frente a la tasa de falsos positivos en distintos umbrales de clasificación.
- **AUC (Area Under the Curve):** área bajo la curva ROC. Mide la capacidad del modelo para distinguir entre clases.

| AUC | Interpretación |
|---|---|
| 1.0 | Clasificador perfecto |
| 0.9 – 1.0 | Excelente |
| 0.7 – 0.9 | Bueno |
| 0.5 | Sin capacidad predictiva (aleatorio) |
| < 0.5 | Peor que aleatorio |

- **Cuándo usarla:** útil para comparar modelos y evaluar rendimiento en datasets desbalanceados, independientemente del umbral de clasificación elegido.

---

### 5. ¿Qué es el feature engineering y feature selection en clasificación?

#### Feature Engineering

Es el proceso de **crear, transformar o combinar variables** existentes para generar nuevas features que mejoren el rendimiento del modelo. Es una de las etapas más creativas y de mayor impacto en el pipeline de Machine Learning.

Técnicas comunes:
- **Codificación de variables categóricas:** One-Hot Encoding, Label Encoding.
- **Escalado de variables numéricas:** normalización (MinMaxScaler), estandarización (StandardScaler).
- **Creación de nuevas variables:** por ejemplo, a partir de una fecha extraer el día de la semana, el mes, si es festivo, etc.
- **Transformaciones matemáticas:** logaritmos, raíces cuadradas para corregir distribuciones sesgadas.
- **Interacciones entre variables:** combinación de dos features para capturar relaciones (ej. precio × superficie).

**Objetivo:** representar los datos de forma que el algoritmo pueda aprender patrones con mayor facilidad y precisión.

#### Feature Selection

Es el proceso de **seleccionar el subconjunto de variables más relevantes** para el modelo, eliminando aquellas que son redundantes, irrelevantes o que introducen ruido.

Métodos principales:

- **Métodos de filtro (Filter):** evalúan la relevancia de cada feature de forma independiente usando estadísticos como la correlación, chi-cuadrado o información mutua.
- **Métodos envolventes (Wrapper):** prueban subconjuntos de features entrenando modelos iterativamente. Ejemplo: Recursive Feature Elimination (RFE).
- **Métodos embebidos (Embedded):** la selección ocurre durante el entrenamiento del modelo. Ejemplo: regularización L1 (Lasso) en regresión logística, importancia de features en árboles.

**Beneficios:**
- Reducción del tiempo de entrenamiento.
- Menor riesgo de overfitting.
- Modelos más simples e interpretables.
- Mejora del rendimiento general al eliminar ruido.

---

### 7. ¿Qué son los hiperparámetros y cómo se optimizan/ajustan?

#### ¿Qué son los hiperparámetros?

Los **hiperparámetros** son parámetros de configuración del modelo que se establecen **antes del entrenamiento** y no se aprenden de los datos. Controlan el proceso de aprendizaje y la estructura del modelo.

Se diferencian de los **parámetros** del modelo (como los pesos en una regresión), que sí son aprendidos automáticamente durante el entrenamiento.

Ejemplos de hiperparámetros por algoritmo:

| Algoritmo | Hiperparámetros |
|---|---|
| Árbol de Decisión | `max_depth`, `min_samples_split`, `criterion` |
| KNN | `n_neighbors`, `metric`, `weights` |
| SVM | `C`, `kernel`, `gamma` |
| Random Forest | `n_estimators`, `max_depth`, `max_features` |
| Regresión Logística | `C`, `solver`, `max_iter` |

#### Optimización de Hiperparámetros (Hyperparameter Tuning)

El proceso de encontrar la combinación óptima de hiperparámetros se denomina **hyperparameter tuning**. Los métodos más utilizados son:

---

**Grid Search (Búsqueda en Rejilla)**

Evalúa **todas las combinaciones posibles** de hiperparámetros definidas en una cuadrícula (grid) predefinida por el usuario.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

- **Ventaja:** garantiza encontrar la mejor combinación dentro del espacio definido.
- **Desventaja:** computacionalmente costoso; el tiempo crece exponencialmente con el número de hiperparámetros y valores.

---

**Random Search (Búsqueda Aleatoria)**

En lugar de probar todas las combinaciones, selecciona **combinaciones aleatorias** dentro del espacio de búsqueda durante un número fijo de iteraciones.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(estimator=modelo, param_distributions=param_dist,
                                   n_iter=50, cv=5, random_state=42)
random_search.fit(X_train, y_train)
print(random_search.best_params_)
```

- **Ventaja:** mucho más eficiente que Grid Search; en la práctica suele encontrar soluciones igual de buenas en menos tiempo.
- **Desventaja:** no garantiza explorar todas las combinaciones posibles.

---

**Comparativa Grid Search vs Random Search**

| Aspecto | Grid Search | Random Search |
|---|---|---|
| Exhaustividad | Evalúa todo el grid | Muestrea aleatoriamente |
| Tiempo de cómputo | Alto | Bajo |
| Garantía de óptimo | Dentro del grid definido | No garantizada |
| Recomendado cuando | Espacio pequeño de búsqueda | Espacio grande de búsqueda |

Ambos métodos utilizan **validación cruzada (cross-validation)** para evaluar cada combinación de hiperparámetros, lo que asegura que los resultados sean robustos y no dependan de una sola partición de los datos.

---