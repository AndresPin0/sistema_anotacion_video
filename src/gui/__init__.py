"""
Módulo de Interfaz Gráfica de Usuario
====================================

Este módulo contiene la interfaz gráfica del sistema de anotación de video.

Principales componentes:
- MainWindow: Ventana principal de la aplicación
- VideoThread: Hilo para procesamiento de video en tiempo real
- ThresholdsDialog: Diálogo para ajustar umbrales
- ActivityHistoryWidget: Widget para mostrar historial de actividades
"""

from .gui_app import MainWindow, main

__all__ = ['MainWindow', 'main'] 