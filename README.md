# Sistema de Anotación de Video

Sistema inteligente de análisis de movimiento corporal en tiempo real capaz de detectar y clasificar actividades específicas de una persona mediante visión por computadora.

## Funcionalidades

- **Detección en tiempo real** de las siguientes actividades:
  - Caminar hacia la cámara
  - Caminar alejándose de la cámara
  - Girar 90 grados
  - Girar 180 grados
  - Sentarse
  - Ponerse de pie

- **Análisis postural** con seguimiento de articulaciones y cálculo de ángulos corporales
- **Interfaz gráfica moderna** con visualización de métricas en tiempo real
- **Historial visual** de actividades detectadas
- **Ajuste de sensibilidad** para personalizar los umbrales de detección
- **Guías visuales** para ayudar al usuario a realizar cada movimiento

## Tecnologías

- **Python**: Lenguaje principal de desarrollo
- **MediaPipe**: Biblioteca de Google para detección de poses
- **OpenCV**: Procesamiento de video y visualización
- **PyQt5**: Interfaz gráfica de usuario
- **NumPy**: Procesamiento numérico y análisis

## Instalación

1. Clonar el repositorio:
   ```
   git clone https://github.com/yourusername/sistema_anotacion_video.git
   cd sistema_anotacion_video
   ```

2. Crear y activar entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Ejecutar la aplicación:
   ```
   python main.py
   ```

2. Seleccionar la cámara cuando se solicite
3. Posicionarse frente a la cámara a una distancia apropiada (2-3 metros)
4. Realizar los movimientos para que sean detectados
5. Usar los botones de prueba para ver ejemplos de cada actividad
6. Ajustar los umbrales de detección según sea necesario usando el botón "Ajustar Umbrales"

## Características técnicas

### Detección de poses
El sistema utiliza MediaPipe para detectar 13 puntos clave del cuerpo humano y calcular las posiciones 3D normalizadas. Estos puntos incluyen:
- Nariz
- Hombros (izquierdo y derecho)
- Codos (izquierdo y derecho)
- Muñecas (izquierda y derecha) 
- Caderas (izquierda y derecha)
- Rodillas (izquierda y derecha)
- Tobillos (izquierdo y derecho)

### Clasificación de actividades
La clasificación se realiza mediante un algoritmo basado en reglas que analiza:
- Posición Z (profundidad) para movimientos hacia/desde la cámara
- Cambios de orientación en los hombros para detectar giros
- Ángulos de rodilla y movimientos de cadera para sentarse/levantarse

### Ajuste de sensibilidad
Los umbrales de detección pueden ajustarse para adaptar el sistema a diferentes condiciones:
- Umbral Z: Sensibilidad al movimiento hacia/desde la cámara
- Umbral de orientación: Sensibilidad a giros
- Umbral de ángulo de rodilla: Flexión necesaria para detectar sentarse/levantarse
- Umbral de movimiento de cadera: Desplazamiento vertical para sentarse/levantarse

## Requisitos del sistema

- Python 3.8 o superior
- Cámara web o dispositivo de captura de video
- Espacio suficiente para realizar los movimientos (aproximadamente 3x3 metros)
- Sistema operativo: Windows 10+, macOS 10.14+, o Linux
