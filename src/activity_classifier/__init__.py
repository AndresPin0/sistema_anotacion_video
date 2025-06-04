"""
Módulo de Clasificación de Actividades
=====================================

Este módulo contiene las clases para clasificar actividades basándose en los
landmarks detectados por MediaPipe.

Principales componentes:
- SimpleActivityClassifier: Clasificador basado en reglas
- MLActivityClassifier: Clasificador usando modelos entrenados
"""

from .activity_classifier import SimpleActivityClassifier
from .ml_activity_classifier import MLActivityClassifier

__all__ = ['SimpleActivityClassifier', 'MLActivityClassifier'] 