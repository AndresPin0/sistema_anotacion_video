"""
Sistema de Anotación de Video
============================

Un sistema completo para la detección y clasificación de actividades humanas en tiempo real
utilizando MediaPipe y técnicas de aprendizaje automático.

Módulos principales:
- pose_detection: Detección de poses y landmarks usando MediaPipe
- activity_classifier: Clasificación de actividades basada en características
- gui: Interfaz gráfica de usuario
- video_capture: Captura y manejo de video
- utils: Utilidades y herramientas auxiliares
"""

__version__ = "1.0.0"
__author__ = "Proyecto Final IA1 - Andrés Pino / Jhonatan Castaño"

from .pose_detection.pose_detector import PoseDetector
from .activity_classifier.activity_classifier import SimpleActivityClassifier
from .video_capture.video_capture import VideoCapture

__all__ = [
    'PoseDetector',
    'SimpleActivityClassifier', 
    'VideoCapture'
]
