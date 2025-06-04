# Sistema de Anotación de Video

**Proyecto Final - Inteligencia Artificial 1**

Un sistema completo para la detección y clasificación de actividades humanas en tiempo real utilizando MediaPipe y técnicas de aprendizaje automático.

## 📋 Descripción del Proyecto

Este sistema permite analizar videos en tiempo real para detectar y clasificar las siguientes actividades:

- **Caminar hacia la cámara**: Persona se acerca frontalmente
- **Caminar alejándose**: Persona se aleja de espaldas
- **Girar 90 grados**: Un cuarto de vuelta a la izquierda o derecha
- **Girar 180 grados**: Media vuelta completa
- **Sentarse**: Desde de pie a sentado en silla
- **Ponerse de pie**: Desde sentado a de pie

### Características Principales

- **Detección de poses en tiempo real** usando MediaPipe
- **Clasificación de actividades** con múltiples algoritmos de ML
- **Análisis de inclinaciones laterales** y movimientos articulares
- **Interfaz gráfica intuitiva** para visualización en tiempo real
- **Pipeline completo** desde extracción hasta entrenamiento
- **Análisis exploratorio** de datos con visualizaciones

## 🏗️ Arquitectura del Sistema

```
sistema_anotacion_video/
├── src/                          # Código fuente principal
│   ├── __init__.py              # Configuración del módulo
│   ├── main.py                  # Punto de entrada principal
│   ├── pose_detection/          # Detección de poses con MediaPipe
│   ├── activity_classifier/     # Clasificación de actividades
│   ├── gui/                     # Interfaz gráfica de usuario
│   ├── video_capture/          # Captura y manejo de video
│   └── utils/                   # Utilidades auxiliares
├── videos/                      # Videos de entrenamiento por actividad
├── data/                        # Datos procesados y características
├── models/                      # Modelos entrenados y evaluaciones
├── analysis/                    # Análisis exploratorio de datos
├── extract_video_features.py   # Extracción de características
├── train_classifier.py         # Entrenamiento de modelos
├── data_analysis.py            # Análisis exploratorio
├── run_pipeline.py             # Pipeline completo automatizado
└── requirements.txt            # Dependencias del proyecto
```

## 🚀 Instalación y Configuración

### Prerequisitos

- Python 3.8 o superior
- Cámara web funcional
- Sistema operativo: Windows, macOS, o Linux

### Instalación

1. **Clonar el repositorio**:
```bash
git clone <url-del-repositorio>
cd sistema_anotacion_video
```

2. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Verificar instalación**:
```bash
python src/main.py
```

## 📹 Preparación de Datos

### Estructura de Videos

Los videos de entrenamiento deben organizarse en la siguiente estructura:

```
videos/
├── caminar_hacia/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── caminar_regreso/
├── girar_90/
├── girar_180/
├── sentarse/
└── ponerse_de_pie/
```

### Recomendaciones para Grabación

✅ **Hacer**:
- Persona completa visible en el frame
- Buena iluminación (evitar sombras fuertes)
- Fondo despejado
- Cámara estable
- Duración: 5-15 segundos por video
- Mínimo 3 videos por actividad

❌ **Evitar**:
- Personas parcialmente cortadas
- Iluminación muy pobre
- Movimientos muy rápidos
- Múltiples personas en el frame

## 🔄 Pipeline de Entrenamiento

### Opción 1: Pipeline Automatizado (Recomendado)

Ejecutar todo el proceso de una vez:

```bash
# Pipeline completo
python run_pipeline.py

# Pipeline rápido (sin Grid Search)
python run_pipeline.py --quick_run

# Para pruebas con pocos datos
python run_pipeline.py --max_videos 2 --max_frames 100 --quick_run
```

### Opción 2: Ejecución Manual por Pasos

#### Paso 1: Extracción de Características
```bash
python extract_video_features.py
```

#### Paso 2: Análisis Exploratorio
```bash
python data_analysis.py
```

#### Paso 3: Entrenamiento de Modelos
```bash
python train_classifier.py
```

## 🎯 Uso del Sistema

### Aplicación Principal

Ejecutar la interfaz gráfica:
```bash
python src/main.py
```

### Funcionalidades de la GUI

- **Vista en tiempo real** de la cámara con detección de poses
- **Clasificación automática** de actividades
- **Visualización de ángulos articulares** y inclinaciones
- **Ajuste de umbrales** de detección en tiempo real
- **Historial de actividades** detectadas
- **Pistas visuales** para guiar al usuario

## 📊 Análisis y Resultados

### Análisis Exploratorio

Los resultados del análisis se guardan en:
- `analysis/reports/analysis_report.txt` - Reporte completo
- `analysis/plots/` - Visualizaciones y gráficos
- `analysis/statistics/` - Estadísticas detalladas

### Evaluación de Modelos

Los resultados del entrenamiento se encuentran en:
- `models/evaluation/model_comparison.csv` - Comparación de modelos
- `models/evaluation/best_model_info.json` - Información del mejor modelo
- `models/plots/` - Gráficos de evaluación

### Métricas Principales

El sistema evalúa los modelos usando:
- **Accuracy**: Precisión general
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precisión y recall

## 🔧 Configuración Avanzada

### Modelos Disponibles

El sistema entrena múltiples algoritmos:
- **Random Forest**: Robusto y rápido
- **SVM**: Efectivo para datos de alta dimensión
- **Logistic Regression**: Simple y interpretable
- **Gradient Boosting**: Potente para patrones complejos
- **K-Nearest Neighbors**: Bueno para decisiones locales
- **XGBoost**: Opcional, requiere instalación adicional

### Personalización de Parámetros

Puedes ajustar parámetros en los scripts:

```bash
# Seleccionar número de características
python train_classifier.py --n_features 30

# Cambiar validation folds
python train_classifier.py --cv_folds 3

# Usar archivo específico de datos
python train_classifier.py --data_file mi_dataset.csv
```

## 📈 Metodología CRISP-DM

El proyecto sigue la metodología CRISP-DM:

1. **Comprensión del Negocio**: Detección de actividades humanas
2. **Comprensión de Datos**: Análisis de videos y landmarks
3. **Preparación de Datos**: Extracción y normalización de características
4. **Modelado**: Entrenamiento de múltiples algoritmos
5. **Evaluación**: Métricas de rendimiento y validación cruzada
6. **Despliegue**: Aplicación en tiempo real

## 🤖 Aspectos Técnicos

### Detección de Poses

- **MediaPipe Pose**: Detección de 33 landmarks corporales
- **Normalización**: Coordenadas relativas al centro de cadera
- **Filtrado**: Suavizado de landmarks para reducir ruido

### Extracción de Características

- **Coordenadas de landmarks**: x, y, z normalizadas
- **Ángulos articulares**: Rodillas, caderas, codos
- **Características geométricas**: Distancias y proporciones
- **Características temporales**: Velocidades y aceleraciones

### Clasificación

- **Enfoque híbrido**: Reglas + Machine Learning
- **Características seleccionadas**: Top N más relevantes
- **Validación cruzada**: Evaluación robusta
- **Optimización**: Grid Search para hiperparámetros

## 🔍 Resolución de Problemas

### Problemas Comunes

1. **Error de cámara**:
   - Verificar que la cámara no esté en uso
   - Probar con diferentes índices de cámara

2. **Baja precisión**:
   - Agregar más videos de entrenamiento
   - Mejorar calidad de iluminación
   - Ajustar umbrales de detección

3. **Rendimiento lento**:
   - Reducir resolución de video
   - Usar modo rápido (--quick_run)
   - Cerrar otras aplicaciones

### Logs y Depuración

- Los logs se muestran en consola durante la ejecución
- Los errores se guardan en archivos de salida
- Usar `--verbose` para más información detallada

## 🤝 Contribución

### Desarrollo

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Implementar cambios con tests
4. Hacer pull request

### Estructura de Código

- **Documentación**: Docstrings en todas las funciones
- **Estilo**: Seguir PEP 8
- **Testing**: Agregar tests para nuevas funcionalidades

## 📚 Referencias y Bibliografia

### Tecnologías Utilizadas

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) - Detección de poses
- [OpenCV](https://opencv.org/) - Procesamiento de video
- [scikit-learn](https://scikit-learn.org/) - Machine Learning
- [PyQt5](https://pypi.org/project/PyQt5/) - Interfaz gráfica

### Artículos de Referencia

- Pose Detection and Activity Recognition using MediaPipe
- Human Activity Recognition: A Comprehensive Survey
- Real-time Activity Classification in Video Streams

## 📄 Licencia

Este proyecto se desarrolla como parte del curso de Inteligencia Artificial 1.

---

## 👥 Equipo de Desarrollo

**Proyecto Final - IA1**
- Análisis de actividades humanas en tiempo real
- Sistema completo de detección y clasificación
- Interfaz gráfica para uso interactivo

---

*Para más información o soporte, consultar la documentación en el código fuente o contactar al equipo de desarrollo.*
