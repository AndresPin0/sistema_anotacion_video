import sys
import os
import cv2
import time
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QComboBox, QRadioButton, QButtonGroup,
                           QFileDialog, QLineEdit, QTextEdit, QGroupBox, QSplitter, QDialog,
                           QFormLayout, QTabWidget, QMessageBox, QSlider, QDoubleSpinBox,
                           QProgressBar, QScrollArea, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QRect, QSize

from ..video_capture.video_capture import VideoCapture
from ..pose_detection.pose_detector import PoseDetector
from ..activity_classifier import SimpleActivityClassifier, MLActivityClassifier

class VideoThread(QThread):
    """Hilo para procesar video en tiempo real"""
    update_frame = pyqtSignal(np.ndarray)
    update_data = pyqtSignal(dict)
    update_activity = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.video_capture = VideoCapture(camera_id)
        self.pose_detector = PoseDetector()
        
        # Configuración de clasificadores
        self.use_ml_classifier = False
        self.simple_classifier = SimpleActivityClassifier()
        self.ml_classifier = None
        self.activity_classifier = self.simple_classifier
        
        # Intentar cargar el clasificador ML
        try:
            self.ml_classifier = MLActivityClassifier()
            print("✅ Clasificador ML cargado exitosamente")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el clasificador ML: {e}")
            print("   Usando clasificador simple por defecto")
        
        self.running = False
        
        # Contador para reducir la frecuencia de actualización de actividad
        self.activity_update_counter = 0
        self.activity_update_frequency = 10  # Actualizar cada 10 frames (~0.33 segundos a 30fps)
        
        # Filtro de actividades para evitar cambios muy rápidos
        self.last_reported_activity = None
        self.activity_confirmation_count = 0
        self.activity_confirmation_threshold = 3  # Confirmar actividad después de 3 detecciones consecutivas
        self.no_activity_counter = 0  # Contador para detectar cuando no hay actividad
        
        # Estado para visualización de pistas
        self.show_movement_hints = False
        self.movement_hint_activity = None
    
    def switch_classifier(self, use_ml=False):
        """Cambia entre el clasificador simple y el ML"""
        if use_ml and self.ml_classifier:
            self.use_ml_classifier = True
            self.activity_classifier = self.ml_classifier
            print("🧠 Clasificador ML activado")
            return True
        else:
            self.use_ml_classifier = False
            self.activity_classifier = self.simple_classifier
            print("📐 Clasificador simple activado")
            return True
        
    def get_classifier_type(self):
        """Retorna el tipo de clasificador actual"""
        return "ML" if self.use_ml_classifier else "Simple"
    
    def run(self):
        """Método principal del hilo - captura de cámara en vivo"""
        try:
            self.video_capture.start_camera()
            self.running = True
            
            while self.running:
                ret, frame, timestamp = self.video_capture.get_frame()
                if not ret:
                    self.error_occurred.emit("Error al capturar frame de la cámara")
                    break
                
                # Procesar frame con MediaPipe
                results_dict, processed_frame = self.pose_detector.process_frame(frame)
                
                # Añadir pistas de movimiento si está activado
                if self.show_movement_hints:
                    processed_frame = self.add_movement_hints(
                        processed_frame, 
                        results_dict.get("landmarks", None)
                    )
                
                # Emitir señales con los resultados
                self.update_frame.emit(processed_frame)
                if results_dict and results_dict["landmarks"]:
                    self.update_data.emit(results_dict)
                    
                    # Clasificar actividad
                    current_activity = self.activity_classifier.predict(
                        results_dict["landmarks"], 
                        results_dict.get("angles", {})
                    )
                    
                    # Filtrar cambios rápidos de actividad
                    if current_activity == self.last_reported_activity:
                        # Misma actividad, incrementar contador de confirmación
                        self.activity_confirmation_count += 1
                    else:
                        # Nueva actividad, resetear contador
                        self.activity_confirmation_count = 1
                        
                    # Reportar la actividad si:
                    # 1. Ha sido confirmada suficientes veces, O
                    # 2. Es el momento de actualizar según la frecuencia Y hay al menos 1 confirmación
                    should_update = (
                        self.activity_confirmation_count >= self.activity_confirmation_threshold or
                        (self.activity_update_counter % self.activity_update_frequency == 0 and 
                         self.activity_confirmation_count >= 1)
                    )
                    
                    if should_update:
                        self.update_activity.emit(current_activity)
                        self.last_reported_activity = current_activity
                        print(f"🎯 Actividad reportada: {current_activity} (confirmaciones: {self.activity_confirmation_count})")
                        self.no_activity_counter = 0  # Reset counter
                    else:
                        # Debug: mostrar qué se está detectando pero no reportando
                        if self.activity_update_counter % 30 == 0:  # Mostrar cada segundo aprox
                            print(f"🔍 Detectando: {current_activity} (confirmaciones: {self.activity_confirmation_count}/{self.activity_confirmation_threshold})")
                        
                        # Incrementar contador de no actividad si la actividad detectada es "ninguna" o "desconocida"
                        if current_activity in ["ninguna", "desconocida"]:
                            self.no_activity_counter += 1
                            # Si no hay actividad por mucho tiempo, reportar "ninguna"
                            if self.no_activity_counter > 60:  # ~2 segundos sin actividad
                                self.update_activity.emit("ninguna")
                                self.last_reported_activity = "ninguna"
                                self.no_activity_counter = 0
                                print("⭕ Sin actividad detectada - mostrando 'ninguna'")
                    
                    self.activity_update_counter += 1
                
                # Pequeña pausa para no sobrecargar la CPU
                time.sleep(0.01)
                
            # Detener la cámara
            self.video_capture.stop_camera()
                
        except Exception as e:
            self.error_occurred.emit(f"Error en el hilo de video: {str(e)}")
    
    def stop(self):
        """Detiene el hilo"""
        self.running = False
        self.wait()
    
    def set_threshold(self, threshold_name, value):
        """Ajusta un umbral específico del clasificador"""
        if threshold_name in self.activity_classifier.thresholds:
            self.activity_classifier.thresholds[threshold_name] = value
            print(f"Umbral {threshold_name} ajustado a {value}")
            return True
        return False
    
    def get_thresholds(self):
        """Obtiene los umbrales actuales del clasificador"""
        return self.activity_classifier.thresholds

    def set_show_movement_hints(self, show, activity=None):
        """Configura la visualización de pistas de movimiento"""
        self.show_movement_hints = show
        self.movement_hint_activity = activity
    
    def add_movement_hints(self, frame, landmarks=None):
        """Añade pistas visuales para guiar al usuario en la realización de movimientos"""
        if not self.show_movement_hints or self.movement_hint_activity is None:
            return frame
            
        h, w, _ = frame.shape
        activity = self.movement_hint_activity
        
        # Crea una copia para dibujar
        overlay = frame.copy()
        alpha = 0.6  # Transparencia
        
        # Define colores para cada actividad
        activity_color = {
            "caminarHacia": (0, 128, 255),  # Azul
            "caminarRegreso": (0, 192, 255),  # Azul claro
            "girar90": (128, 0, 255),  # Púrpura
            "girar180": (192, 0, 255),  # Púrpura claro
            "sentarse": (0, 255, 128),  # Verde
            "ponerseDePie": (128, 255, 0)   # Verde amarillento
        }
        
        color = activity_color.get(activity, (200, 200, 200))
        
        # Dibujar pistas según la actividad
        if activity in ["caminarHacia", "caminarRegreso"]:
            # Dibujar flechas para caminar
            arrow_start = (w // 2, h // 2)
            if activity == "caminarHacia":
                # Flecha desde el centro hacia la cámara (abajo)
                arrow_end = (w // 2, h // 2 + 100)
                text = "Camina HACIA la cámara"
            else:
                # Flecha desde el centro alejándose de la cámara (arriba)
                arrow_end = (w // 2, h // 2 - 100)
                text = "Camina ALEJÁNDOTE de la cámara"
                
            # Dibujar flecha
            cv2.arrowedLine(overlay, arrow_start, arrow_end, color, 5, tipLength=0.3)
            
        elif activity in ["girar90", "girar180"]:
            # Dibujar arco para girar
            center = (w // 2, h // 2)
            radius = 100
            angle = 90 if activity == "girar90" else 180
            text = f"Gira {angle}°"
            
            # Dibujar círculo
            cv2.circle(overlay, center, radius, color, 2)
            
            # Dibujar flecha del arco
            start_angle = 0
            end_angle = np.radians(angle)
            
            # Punto inicial y final del arco
            start_point = (int(center[0] + radius), center[1])
            end_x = int(center[0] + radius * np.cos(end_angle))
            end_y = int(center[1] - radius * np.sin(end_angle))
            end_point = (end_x, end_y)
            
            # Dibujar líneas del arco
            cv2.line(overlay, center, start_point, color, 2)
            cv2.line(overlay, center, end_point, color, 2)
            
            # Dibujar arco
            cv2.ellipse(overlay, center, (radius, radius), 0, 0, angle, color, 3)
            
        elif activity in ["sentarse", "ponerseDePie"]:
            # Dibujar flechas para sentarse/levantarse
            if activity == "sentarse":
                # Flecha hacia abajo para sentarse
                arrow_start = (w // 2, h // 3)
                arrow_end = (w // 2, 2 * h // 3)
                text = "Siéntate en una silla"
            else:
                # Flecha hacia arriba para ponerse de pie
                arrow_start = (w // 2, 2 * h // 3)
                arrow_end = (w // 2, h // 3)
                text = "Levántate de la silla"
                
            # Dibujar flecha
            cv2.arrowedLine(overlay, arrow_start, arrow_end, color, 5, tipLength=0.3)
            
            # Dibujar silueta de persona
            if landmarks:
                # Si tenemos landmarks, dibujar líneas para indicar flexión de rodillas
                pass
        
        # Mezclar el overlay con el frame original
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Añadir texto instructivo
        cv2.putText(frame, text, (w // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame


class ThresholdsDialog(QDialog):
    """Diálogo para ajustar los umbrales de detección de actividades en tiempo real"""
    
    def __init__(self, video_thread, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajustar Umbrales de Detección")
        self.setMinimumSize(500, 400)
        self.video_thread = video_thread
        
        # Obtener umbrales actuales
        self.current_thresholds = self.video_thread.get_thresholds()
        
        # Crear interfaz
        self._create_ui()
    
    def _create_ui(self):
        """Crea la interfaz del diálogo de umbrales"""
        layout = QVBoxLayout(self)
        
        # Grupo para umbrales de caminar
        walking_group = QGroupBox("Umbrales para Caminar")
        walking_layout = QFormLayout(walking_group)
        
        # Z threshold (movimiento hacia/desde cámara)
        self.z_threshold_spin = QDoubleSpinBox()
        self.z_threshold_spin.setRange(0.001, 0.1)
        self.z_threshold_spin.setSingleStep(0.001)
        self.z_threshold_spin.setDecimals(3)
        self.z_threshold_spin.setValue(self.current_thresholds["z_threshold"])
        self.z_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("z_threshold", value)
        )
        walking_layout.addRow("Umbral Z (caminar):", self.z_threshold_spin)
        
        # Explicación
        walking_layout.addRow(QLabel("Valores más bajos = más sensible a detectar movimiento hacia/desde cámara"))
        
        layout.addWidget(walking_group)
        
        # Grupo para umbrales de giro
        turn_group = QGroupBox("Umbrales para Giros")
        turn_layout = QFormLayout(turn_group)
        
        # Orientation threshold (giro 90°)
        self.orientation_threshold_spin = QDoubleSpinBox()
        self.orientation_threshold_spin.setRange(1.0, 30.0)
        self.orientation_threshold_spin.setSingleStep(0.5)
        self.orientation_threshold_spin.setDecimals(1)
        self.orientation_threshold_spin.setValue(self.current_thresholds["orientation_threshold"])
        self.orientation_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("orientation_threshold", value)
        )
        turn_layout.addRow("Umbral Orientación (giro 90°):", self.orientation_threshold_spin)
        
        # Orientation threshold large (giro 180°)
        self.orientation_threshold_large_spin = QDoubleSpinBox()
        self.orientation_threshold_large_spin.setRange(20.0, 90.0)
        self.orientation_threshold_large_spin.setSingleStep(1.0)
        self.orientation_threshold_large_spin.setDecimals(1)
        self.orientation_threshold_large_spin.setValue(self.current_thresholds["orientation_threshold_large"])
        self.orientation_threshold_large_spin.valueChanged.connect(
            lambda value: self.update_threshold("orientation_threshold_large", value)
        )
        turn_layout.addRow("Umbral Orientación Grande (giro 180°):", self.orientation_threshold_large_spin)
        
        # Explicación
        turn_layout.addRow(QLabel("Valores más bajos = más sensible a detectar giros"))
        
        layout.addWidget(turn_group)
        
        # Grupo para umbrales de sentarse/levantarse
        sit_group = QGroupBox("Umbrales para Sentarse/Levantarse")
        sit_layout = QFormLayout(sit_group)
        
        # Knee angle threshold
        self.knee_angle_threshold_spin = QDoubleSpinBox()
        self.knee_angle_threshold_spin.setRange(100.0, 180.0)
        self.knee_angle_threshold_spin.setSingleStep(1.0)
        self.knee_angle_threshold_spin.setDecimals(1)
        self.knee_angle_threshold_spin.setValue(self.current_thresholds["knee_angle_threshold"])
        self.knee_angle_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("knee_angle_threshold", value)
        )
        sit_layout.addRow("Umbral Ángulo Rodilla:", self.knee_angle_threshold_spin)
        
        # Hip movement threshold
        self.hip_movement_threshold_spin = QDoubleSpinBox()
        self.hip_movement_threshold_spin.setRange(0.0001, 0.01)
        self.hip_movement_threshold_spin.setSingleStep(0.0001)
        self.hip_movement_threshold_spin.setDecimals(4)
        self.hip_movement_threshold_spin.setValue(self.current_thresholds["hip_movement_threshold"])
        self.hip_movement_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("hip_movement_threshold", value)
        )
        sit_layout.addRow("Umbral Movimiento Cadera:", self.hip_movement_threshold_spin)
        
        # Explicación
        sit_layout.addRow(QLabel("Valores más altos en ángulo = más sensible a flexión de rodilla"))
        sit_layout.addRow(QLabel("Valores más bajos en movimiento = más sensible a movimiento de cadera"))
        
        layout.addWidget(sit_group)
        
        # Botones
        button_layout = QHBoxLayout()
        
        # Botón para restaurar valores por defecto
        defaults_button = QPushButton("Restaurar Valores Predeterminados")
        defaults_button.clicked.connect(self.restore_defaults)
        button_layout.addWidget(defaults_button)
        
        # Espaciador
        button_layout.addStretch()
        
        # Botón para cerrar
        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def update_threshold(self, threshold_name, value):
        """Actualiza un umbral en tiempo real"""
        if self.video_thread.set_threshold(threshold_name, value):
            print(f"Umbral {threshold_name} actualizado a {value}")
    
    def restore_defaults(self):
        """Restaura los valores predeterminados de los umbrales"""
        default_thresholds = {
            "z_threshold": 0.01,
            "knee_angle_threshold": 170,
            "hip_movement_threshold": 0.001,
            "orientation_threshold": 10,
            "orientation_threshold_large": 40
        }
        
        # Actualizar los controles de la interfaz
        self.z_threshold_spin.setValue(default_thresholds["z_threshold"])
        self.knee_angle_threshold_spin.setValue(default_thresholds["knee_angle_threshold"])
        self.hip_movement_threshold_spin.setValue(default_thresholds["hip_movement_threshold"])
        self.orientation_threshold_spin.setValue(default_thresholds["orientation_threshold"])
        self.orientation_threshold_large_spin.setValue(default_thresholds["orientation_threshold_large"])
        
        # Actualizar los umbrales en el clasificador
        for name, value in default_thresholds.items():
            self.video_thread.set_threshold(name, value)
        
        QMessageBox.information(self, "Valores Restaurados", 
                               "Los umbrales han sido restaurados a sus valores predeterminados.")


class ActivityHistoryWidget(QWidget):
    """Widget para mostrar un historial visual de actividades detectadas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(60)
        
        # Historial de actividades (tuplas de actividad y confianza)
        self.activity_history = []
        self.max_history_length = 20
        
        # Mapeo de actividades a colores
        self.activity_colors = {
            "caminarHacia": QColor(0, 128, 255),  # Azul
            "caminarRegreso": QColor(0, 192, 255),  # Azul claro
            "girar90": QColor(128, 0, 255),  # Púrpura
            "girar180": QColor(192, 0, 255),  # Púrpura claro
            "sentarse": QColor(0, 255, 128),  # Verde
            "ponerseDePie": QColor(128, 255, 0),  # Verde amarillento
            "desconocida": QColor(128, 128, 128),  # Gris
            "ninguna": QColor(64, 64, 64)  # Gris oscuro
        }
        
        # Nombres en español para mostrar
        self.activity_names = {
            "caminarHacia": "HACIA",
            "caminarRegreso": "REGRESO",
            "girar90": "GIRO 90°",
            "girar180": "GIRO 180°",
            "sentarse": "SENTARSE",
            "ponerseDePie": "DE PIE",
            "desconocida": "?",
            "ninguna": "-"
        }
    
    def add_activity(self, activity, confidence=0.0):
        """Añade una actividad al historial solo si es diferente a la última"""
        # Solo agregar si es diferente a la última actividad registrada
        if not self.activity_history or self.activity_history[-1][0] != activity:
            self.activity_history.append((activity, confidence))
            if len(self.activity_history) > self.max_history_length:
                self.activity_history.pop(0)
            self.update()
            print(f"Historial actualizado: {activity} (confianza: {confidence:.1f}%)")
    
    def paintEvent(self, event):
        """Dibuja el historial de actividades"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dibujar fondo
        painter.fillRect(event.rect(), QBrush(QColor(30, 30, 30)))
        
        # Si no hay datos, mostrar mensaje
        if not self.activity_history:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(event.rect(), Qt.AlignCenter, "No hay actividades registradas")
            return
        
        # Calcular tamaño de cada bloque de actividad
        width = self.width()
        height = self.height()
        block_width = width / min(self.max_history_length, max(1, len(self.activity_history)))
        
        # Dibujar cada actividad
        x = 0
        for i, (activity, confidence) in enumerate(self.activity_history):
            # Color según actividad
            color = self.activity_colors.get(activity, QColor(100, 100, 100))
            
            # Dibujar bloque con altura según confianza
            block_height = height * min(1.0, max(0.2, confidence / 100))
            y = height - block_height
            
            # Dibujar rectángulo - Convertir coordenadas a enteros
            painter.fillRect(QRect(int(x), int(y), int(block_width), int(block_height)), QBrush(color))
            
            # Texto de actividad
            painter.setPen(Qt.white)
            activity_text = self.activity_names.get(activity, activity)
            text_rect = QRect(int(x), int(y), int(block_width), int(block_height))
            painter.drawText(text_rect, Qt.AlignCenter, activity_text)
            
            # Línea divisoria
            painter.setPen(QPen(QColor(50, 50, 50), 1))
            painter.drawLine(int(x + block_width), 0, int(x + block_width), height)
            
            # Avanzar posición
            x += block_width
        
        # Dibujar bordes
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(0, 0, width - 1, height - 1)
        
        # Escala de tiempo en la parte inferior
        painter.setPen(Qt.white)
        painter.drawText(QRect(5, height - 20, 50, 15), Qt.AlignLeft, "Anterior")
        painter.drawText(QRect(width - 55, height - 20, 50, 15), Qt.AlignRight, "Actual")


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Sistema de Anotación de Video - Análisis de Actividades")
        self.setMinimumSize(1280, 800)
        
        # Estilo global de la aplicación
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QGroupBox {
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #3498db;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c5a85;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QLabel {
                color: #ecf0f1;
            }
            QTextEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
            }
        """)
        
        # Inicializar variables
        self.video_thread = VideoThread()
        self.video_thread.update_frame.connect(self.update_video_frame)
        self.video_thread.update_data.connect(self.update_data_display)
        self.video_thread.update_activity.connect(self.update_activity_display)
        self.video_thread.error_occurred.connect(self.show_error)
        
        self.recording = False
        self.current_activity = "ninguna"
        self.auto_detect_activity = True  # Flag para habilitar/deshabilitar detección automática
        
        # Configurar la interfaz
        self._create_ui()
        
        # Mostrar mensaje de inicio
        QTimer.singleShot(500, self.show_welcome_message)
    
    def show_welcome_message(self):
        """Muestra un mensaje de bienvenida con instrucciones básicas"""
        QMessageBox.information(
            self,
            "Bienvenido al Sistema de Anotación de Video",
            "Este sistema permite detectar automáticamente actividades como:\n\n"
            "• Caminar hacia/alejándose de la cámara\n"
            "• Girar 90° o 180°\n"
            "• Sentarse y ponerse de pie\n\n"
            "Para comenzar:\n"
            "1. Presione 'Iniciar Cámara'\n"
            "2. Colóquese frente a la cámara a una distancia adecuada\n"
            "3. Realice los movimientos para que sean detectados\n\n"
            "Puede probar cada actividad con los botones de prueba o ajustar los\n"
            "umbrales de detección para mejorar la precisión."
        )
    
    def _create_ui(self):
        """Crea la interfaz de usuario simplificada sin pestañas"""
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)  # Añadir más espacio entre paneles
        
        # Panel izquierdo (video)
        video_panel = QVBoxLayout()
        video_panel.setSpacing(8)  # Espaciado entre elementos del panel
        
        # Título del panel de video
        video_title = QLabel("Vista de Cámara y Detección de Actividades")
        video_title.setAlignment(Qt.AlignCenter)
        video_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px; color: #3498db;")
        video_panel.addWidget(video_title)
        
        # Contenedor para el video y la etiqueta de actividad
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container.setStyleSheet("background-color: black; border-radius: 5px;")
        
        # Etiqueta para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: transparent;")
        video_container_layout.addWidget(self.video_label)
        
        # Etiqueta para mostrar la actividad actual (superpuesta)
        self.activity_label = QLabel("ACTIVIDAD: ninguna")
        self.activity_label.setAlignment(Qt.AlignCenter)
        self.activity_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180); 
            color: white; 
            font-size: 24px; 
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        """)
        self.activity_label.setFixedHeight(60)
        video_container_layout.addWidget(self.activity_label)
        
        video_panel.addWidget(video_container)
        
        # Título del historial
        history_title = QLabel("Historial de Actividades")
        history_title.setAlignment(Qt.AlignCenter)
        history_title.setStyleSheet("font-size: 14px; margin-top: 5px; color: #3498db;")
        video_panel.addWidget(history_title)
        
        # Widget de historial de actividades
        self.history_widget = ActivityHistoryWidget()
        video_panel.addWidget(self.history_widget)
        
        # Panel de controles de actividad
        controls_group = QGroupBox("Controles de Actividad")
        controls_layout = QVBoxLayout(controls_group)
        
        # Panel de prueba de actividades
        test_panel = QHBoxLayout()
        test_panel.addWidget(QLabel("Prueba de Actividades:"))
        
        # Botones para forzar cada actividad (modo de prueba) - mejor estilo
        button_layout = QHBoxLayout()
        self.test_buttons = {}
        
        # Primera fila de botones
        walking_layout = QHBoxLayout()
        walking_title = QLabel("Caminar:")
        walking_title.setFixedWidth(80)
        walking_layout.addWidget(walking_title)
        
        for activity in ["caminarHacia", "caminarRegreso"]:
            btn = QPushButton(activity.capitalize().replace("caminar", ""))
            btn.setFixedHeight(35)
            if activity == "caminarHacia":
                btn.setStyleSheet("background-color: #3498db;")
            else:
                btn.setStyleSheet("background-color: #2980b9;")
            btn.clicked.connect(lambda checked, act=activity: self.test_activity(act))
            walking_layout.addWidget(btn)
            self.test_buttons[activity] = btn
        
        controls_layout.addLayout(walking_layout)
        
        # Segunda fila de botones
        turn_layout = QHBoxLayout()
        turn_title = QLabel("Girar:")
        turn_title.setFixedWidth(80)
        turn_layout.addWidget(turn_title)
        
        for activity in ["girar90", "girar180"]:
            btn = QPushButton(activity.capitalize().replace("girar", ""))
            btn.setFixedHeight(35)
            if activity == "girar90":
                btn.setStyleSheet("background-color: #9b59b6;")
            else:
                btn.setStyleSheet("background-color: #8e44ad;")
            btn.clicked.connect(lambda checked, act=activity: self.test_activity(act))
            turn_layout.addWidget(btn)
            self.test_buttons[activity] = btn
        
        controls_layout.addLayout(turn_layout)
        
        # Tercera fila de botones
        sit_layout = QHBoxLayout()
        sit_title = QLabel("Postura:")
        sit_title.setFixedWidth(80)
        sit_layout.addWidget(sit_title)
        
        for activity in ["sentarse", "ponerseDePie"]:
            btn = QPushButton(activity.capitalize())
            btn.setFixedHeight(35)
            if activity == "sentarse":
                btn.setStyleSheet("background-color: #2ecc71;")
            else:
                btn.setStyleSheet("background-color: #27ae60;")
            btn.clicked.connect(lambda checked, act=activity: self.test_activity(act))
            sit_layout.addWidget(btn)
            self.test_buttons[activity] = btn
        
        controls_layout.addLayout(sit_layout)
        
        video_panel.addWidget(controls_group)
        
        # Controles de cámara
        camera_controls = QVBoxLayout()
        
        # Primera fila de controles
        first_row = QHBoxLayout()
        
        # Botón para iniciar/detener cámara
        self.camera_button = QPushButton("Iniciar Cámara")
        self.camera_button.setFixedHeight(40)
        self.camera_button.setStyleSheet("background-color: #e74c3c;")
        self.camera_button.clicked.connect(self.toggle_camera)
        first_row.addWidget(self.camera_button)
        
        # Botón para habilitar/deshabilitar detección automática
        self.auto_detect_button = QPushButton("Detección Auto: ON")
        self.auto_detect_button.setFixedHeight(40)
        self.auto_detect_button.setCheckable(True)
        self.auto_detect_button.setChecked(True)
        self.auto_detect_button.setStyleSheet("""
            QPushButton:checked { background-color: #2ecc71; }
            QPushButton:!checked { background-color: #e74c3c; }
        """)
        self.auto_detect_button.clicked.connect(self.toggle_auto_detection)
        first_row.addWidget(self.auto_detect_button)
        
        camera_controls.addLayout(first_row)
        
        # Segunda fila de controles
        second_row = QHBoxLayout()
        
        # Selector de clasificador
        classifier_container = QHBoxLayout()
        classifier_label = QLabel("Clasificador:")
        classifier_container.addWidget(classifier_label)
        
        self.classifier_selector = QComboBox()
        self.classifier_selector.addItems(["Simple (Reglas)", "ML (Entrenado)"])
        self.classifier_selector.setFixedHeight(40)
        self.classifier_selector.currentTextChanged.connect(self.change_classifier)
        classifier_container.addWidget(self.classifier_selector)
        
        second_row.addLayout(classifier_container)
        
        # Botón para ajustar umbrales
        self.adjust_thresholds_button = QPushButton("Ajustar Umbrales")
        self.adjust_thresholds_button.setFixedHeight(40)
        self.adjust_thresholds_button.setStyleSheet("background-color: #f39c12;")
        self.adjust_thresholds_button.clicked.connect(self.show_thresholds_dialog)
        second_row.addWidget(self.adjust_thresholds_button)
        
        camera_controls.addLayout(second_row)
        
        video_panel.addLayout(camera_controls)
        
        # Panel derecho (datos)
        data_panel = QVBoxLayout()
        data_panel.setSpacing(10)  # Espaciado entre elementos del panel
        
        # Título del panel de datos
        data_title = QLabel("Datos y Métricas")
        data_title.setAlignment(Qt.AlignCenter)
        data_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px; color: #3498db;")
        data_panel.addWidget(data_title)
        
        # Grupo para actividad actual y métricas clave
        activity_group = QGroupBox("Actividad Actual y Métricas")
        activity_layout = QVBoxLayout(activity_group)
        
        # Métrica para estado de movimiento
        self.activity_metrics = {}
        metrics = [
            ("actividad_actual", "Actividad Actual: --"),
            ("confianza", "Confianza: --"),
            ("estado_movimiento", "Estado de Movimiento: --"),
            ("orientacion", "Orientación: --"),
            ("angulo_rodillas", "Ángulo Rodillas: --"),
            ("inclinacion_tronco", "Inclinación Tronco: --")
        ]
        
        for key, text in metrics:
            if key == "confianza":
                # Para la confianza, crear una disposición horizontal con etiqueta y barra de progreso
                confidence_layout = QHBoxLayout()
                
                # Etiqueta de confianza
                label = QLabel(text)
                label.setStyleSheet("font-size: 14px;")
                confidence_layout.addWidget(label, 2)  # Proporción 2:3 para etiqueta:barra
                
                # Barra de progreso para confianza
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                progress_bar.setValue(0)
                progress_bar.setFormat("%v%")
                progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #555;
                        border-radius: 3px;
                        text-align: center;
                        background-color: #333;
                        height: 20px;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #27ae60);
                        border-radius: 2px;
                    }
                """)
                confidence_layout.addWidget(progress_bar, 3)
                
                # Guardar ambos elementos
                self.activity_metrics[key] = label
                self.activity_metrics[key + "_progress"] = progress_bar
                
                # Añadir el layout horizontal al layout principal
                activity_layout.addLayout(confidence_layout)
            else:
                # Para otras métricas, solo la etiqueta
                label = QLabel(text)
                label.setStyleSheet("font-size: 14px;")
                activity_layout.addWidget(label)
                self.activity_metrics[key] = label
        
        # Añadir indicadores visuales de confianza para cada actividad
        confidence_group = QGroupBox("Confianza por Actividad")
        confidence_layout = QVBoxLayout(confidence_group)
        
        # Crear barras de progreso para cada actividad
        self.activity_confidence_bars = {}
        for activity in ["caminarHacia", "caminarRegreso", "girar90", "girar180", "sentarse", "ponerseDePie"]:
            # Crear layout horizontal para cada actividad
            activity_bar_layout = QHBoxLayout()
            
            # Nombre de la actividad
            label = QLabel(activity.capitalize())
            label.setFixedWidth(120)
            activity_bar_layout.addWidget(label)
            
            # Barra de progreso
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%v%")
            
            # Estilo según el tipo de actividad
            if activity in ["caminarHacia", "caminarRegreso"]:
                color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #1a5276)"  # Azul
            elif activity in ["girar90", "girar180"]:
                color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9b59b6, stop:1 #6c3483)"  # Púrpura
            else:  # sentarse, ponerseDePie
                color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ecc71, stop:1 #1e8449)"  # Verde
            
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #333;
                    height: 15px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 2px;
                }}
            """)
            
            activity_bar_layout.addWidget(progress_bar)
            
            # Guardar la barra de progreso
            self.activity_confidence_bars[activity] = progress_bar
            
            # Añadir al layout
            confidence_layout.addLayout(activity_bar_layout)
        
        data_panel.addWidget(activity_group)
        data_panel.addWidget(confidence_group)
        
        # Grupo para datos de ángulos
        angles_group = QGroupBox("Ángulos Articulares")
        angles_layout = QVBoxLayout(angles_group)
        
        # Crear etiquetas para cada ángulo
        self.angle_labels = {}
        for angle_name in ["left_knee_angle", "right_knee_angle", "left_hip_angle", 
                          "right_hip_angle", "left_elbow_angle", "right_elbow_angle", 
                          "trunk_lateral_inclination"]:
            label = QLabel(f"{angle_name}: --")
            angles_layout.addWidget(label)
            self.angle_labels[angle_name] = label
        
        data_panel.addWidget(angles_group)
        
        # Grupo para landmarks detectados
        landmarks_group = QGroupBox("Landmarks Detectados")
        landmarks_layout = QVBoxLayout(landmarks_group)
        
        self.landmarks_text = QTextEdit()
        self.landmarks_text.setReadOnly(True)
        landmarks_layout.addWidget(self.landmarks_text)
        
        data_panel.addWidget(landmarks_group)
        
        # Agregar paneles al layout con proporciones 2:1
        main_layout.addLayout(video_panel, 2)
        main_layout.addLayout(data_panel, 1)
    
    def update_video_frame(self, frame):
        """Actualiza el frame de video en la interfaz"""
        # Convertir el frame de OpenCV a QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Redimensionar manteniendo la relación de aspecto
        pixmap = QPixmap.fromImage(qt_image)
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Mostrar en la etiqueta
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_data_display(self, results_dict):
        """Actualiza la visualización de datos"""
        if not results_dict:
            return
        
        # Actualizar ángulos
        angles = results_dict.get("angles", {})
        if angles:
            for angle_name, angle_value in angles.items():
                if angle_name in self.angle_labels:
                    self.angle_labels[angle_name].setText(f"{angle_name}: {angle_value:.1f}°")
            
            # Actualizar métricas claves para la detección de actividad
            if "left_knee_angle" in angles and "right_knee_angle" in angles:
                avg_knee = (angles["left_knee_angle"] + angles["right_knee_angle"]) / 2
                rodillas_estado = "Dobladas" if avg_knee < 150 else "Extendidas"
                self.activity_metrics["angulo_rodillas"].setText(f"Ángulo Rodillas: {avg_knee:.1f}° ({rodillas_estado})")
            
            if "trunk_lateral_inclination" in angles:
                inclinacion = angles["trunk_lateral_inclination"]
                lado = "Izquierda" if inclinacion > 5 else "Derecha" if inclinacion < -5 else "Centro"
                self.activity_metrics["inclinacion_tronco"].setText(f"Inclinación Tronco: {inclinacion:.1f}° ({lado})")
        
        # Actualizar landmarks
        landmarks = results_dict.get("landmarks", {})
        if landmarks:
            # Mostrar un resumen de los landmarks
            summary = "Landmarks detectados:\n"
            
            # Actualizar métricas basadas en landmarks
            if "nose" in landmarks:
                # Mostrar la posición Z de la nariz para indicar hacia dónde se está moviendo
                nose_z = landmarks["nose"]["z"]
                estado = "Hacia la cámara" if nose_z < -0.03 else "Alejándose" if nose_z > 0.03 else "Estático"
                self.activity_metrics["estado_movimiento"].setText(f"Estado de Movimiento: {estado} (Z: {nose_z:.3f})")
            
            # Mostrar detalles de algunos landmarks clave
            for landmark_name in ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
                if landmark_name in landmarks:
                    lm = landmarks[landmark_name]
                    summary += f"{landmark_name}: ({lm['x']:.2f}, {lm['y']:.2f}, {lm['z']:.2f})\n"
            
            self.landmarks_text.setText(summary)
    
    def update_activity_display(self, activity):
        """Actualiza la visualización de la actividad detectada"""
        # Solo actualizar si la detección automática está habilitada
        if self.auto_detect_activity:
            # No mostrar 'desconocida' en la interfaz
            if activity != "desconocida":
                # Solo actualizar si es una actividad diferente a la actual
                if self.current_activity != activity:
                    self.current_activity = activity
                    
                    # Asignar color según la actividad
                    activity_colors = {
                        "caminarHacia": "background-color: rgba(0, 128, 255, 180);",  # Azul
                        "caminarRegreso": "background-color: rgba(0, 192, 255, 180);",  # Azul claro
                        "girar90": "background-color: rgba(128, 0, 255, 180);",  # Púrpura
                        "girar180": "background-color: rgba(192, 0, 255, 180);",  # Púrpura claro
                        "sentarse": "background-color: rgba(0, 255, 128, 180);",  # Verde
                        "ponerseDePie": "background-color: rgba(128, 255, 0, 180);",  # Verde amarillento
                        "ninguna": "background-color: rgba(0, 0, 0, 180);"  # Negro
                    }
                    
                    # Actualizar la métrica de actividad actual
                    self.activity_metrics["actividad_actual"].setText(f"Actividad Actual: {self.current_activity}")
                    
                    # Traducir actividad para mostrar en español
                    actividad_es = {
                        "caminarHacia": "CAMINANDO HACIA CÁMARA",
                        "caminarRegreso": "CAMINANDO ALEJÁNDOSE",
                        "girar90": "GIRANDO 90°",
                        "girar180": "GIRANDO 180°",
                        "sentarse": "SENTÁNDOSE",
                        "ponerseDePie": "PONIÉNDOSE DE PIE",
                        "ninguna": "NINGUNA"
                    }.get(self.current_activity, self.current_activity.upper())
                    
                    # Estilo base
                    base_style = """
                        color: white; 
                        font-size: 24px; 
                        font-weight: bold;
                        border-radius: 10px;
                        padding: 10px;
                    """
                    
                    # Obtener color para la actividad actual
                    color_style = activity_colors.get(activity, activity_colors["ninguna"])
                    
                    # Establecer estilo completo
                    self.activity_label.setStyleSheet(color_style + base_style)
                    self.activity_label.setText(f"ACTIVIDAD: {actividad_es}")
                    
                    # Actualizar confianza (simulada - en realidad es la consistencia de detección)
                    historial = self.video_thread.activity_classifier.prediction_history if hasattr(self.video_thread.activity_classifier, 'prediction_history') else []
                    confianza = 0
                    if historial and len(historial) > 1:
                        actividad_actual_count = historial.count(self.current_activity)
                        confianza = actividad_actual_count / len(historial) * 100
                        nivel = "Alta" if confianza > 80 else "Media" if confianza > 50 else "Baja"
                        self.activity_metrics["confianza"].setText(f"Confianza: {confianza:.0f}% ({nivel})")
                        
                        # Actualizar barra de progreso de confianza
                        if "confianza_progress" in self.activity_metrics:
                            self.activity_metrics["confianza_progress"].setValue(int(confianza))
                    else:
                        # Usar confianza basada en la confirmación
                        confirmation_ratio = self.video_thread.activity_confirmation_count / max(1, self.video_thread.activity_confirmation_threshold)
                        confianza = min(100, confirmation_ratio * 100)
                        nivel = "Alta" if confianza > 80 else "Media" if confianza > 50 else "Baja"
                        self.activity_metrics["confianza"].setText(f"Confianza: {confianza:.0f}% ({nivel})")
                        
                        if "confianza_progress" in self.activity_metrics:
                            self.activity_metrics["confianza_progress"].setValue(int(confianza))
                    
                    # Añadir al historial visual (solo si es una nueva actividad)
                    self.history_widget.add_activity(self.current_activity, confianza)
                    
                    # Actualizar todas las barras de confianza por actividad
                    self._update_activity_confidence_bars(historial)
                    
                    # Imprimir la actividad en la consola (para debug y seguimiento)
                    print(f"ACTIVIDAD DETECTADA: {self.current_activity}")
                else:
                    # Misma actividad, solo actualizar métricas de confianza sin cambiar historial
                    confirmation_ratio = self.video_thread.activity_confirmation_count / max(1, self.video_thread.activity_confirmation_threshold)
                    confianza = min(100, confirmation_ratio * 100)
                    nivel = "Alta" if confianza > 80 else "Media" if confianza > 50 else "Baja"
                    self.activity_metrics["confianza"].setText(f"Confianza: {confianza:.0f}% ({nivel})")
                    
                    if "confianza_progress" in self.activity_metrics:
                        self.activity_metrics["confianza_progress"].setValue(int(confianza))
    
    def toggle_camera(self):
        """Inicia o detiene la cámara"""
        if not self.video_thread.running:
            # Iniciar cámara
            self.video_thread.start()
            self.camera_button.setText("Detener Cámara")
        else:
            # Detener cámara
            self.video_thread.stop()
            self.camera_button.setText("Iniciar Cámara")
            
            # Limpiar la visualización
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: black;")
            
            # Resetear etiquetas de ángulos
            for label in self.angle_labels.values():
                label.setText(label.text().split(":")[0] + ": --")
            
            self.landmarks_text.clear()
            
            # Asegurarse de desactivar las pistas visuales
            self.video_thread.set_show_movement_hints(False)
    
    def toggle_auto_detection(self):
        """Habilita o deshabilita la detección automática de actividades"""
        self.auto_detect_activity = self.auto_detect_button.isChecked()
        button_text = "Detección Auto: ON" if self.auto_detect_activity else "Detección Auto: OFF"
        self.auto_detect_button.setText(button_text)
        print(f"Detección automática de actividad: {'ACTIVADA' if self.auto_detect_activity else 'DESACTIVADA'}")
        
        # Si se activa, mostrar consejos de movimiento
        if self.auto_detect_activity:
            QMessageBox.information(
                self,
                "Detección Activada",
                "La detección automática de actividades está ACTIVADA.\n\n"
                "Consejos para mejorar la detección:\n"
                "• CAMINAR: Muévase claramente hacia/lejos de la cámara\n"
                "• GIRAR: Gire el cuerpo completo a 90° o 180°\n"
                "• SENTARSE: Flexione las rodillas y baje el cuerpo\n"
                "• LEVANTARSE: Desde posición sentada, extienda rodillas\n\n"
                "La sensibilidad ha sido aumentada para detectar mejor los movimientos."
            )
        # Si se deshabilita, mostrar 'ninguna' en la interfaz
        else:
            self.current_activity = "ninguna"
            self.activity_label.setText(f"ACTIVIDAD: {self.current_activity}")
            
            # Limpiar historial visual y barras de confianza
            self.history_widget.activity_history = []
            self.history_widget.update()
            
            for activity in self.activity_confidence_bars:
                self.activity_confidence_bars[activity].setValue(0)
                
            if "confianza_progress" in self.activity_metrics:
                self.activity_metrics["confianza_progress"].setValue(0)
    
    def show_error(self, message):
        """Muestra un mensaje de error"""
        QMessageBox.critical(self, "Error", message)
    
    def closeEvent(self, event):
        """Maneja el evento de cierre de la ventana"""
        # Detener hilo de video si está activo
        if self.video_thread.running:
            self.video_thread.stop()
        event.accept()

    def test_activity(self, activity):
        """Prueba una actividad específica actualizando la interfaz"""
        self.current_activity = activity
        
        # Asignar color según la actividad
        activity_colors = {
            "caminarHacia": "background-color: rgba(0, 128, 255, 180);",  # Azul
            "caminarRegreso": "background-color: rgba(0, 192, 255, 180);",  # Azul claro
            "girar90": "background-color: rgba(128, 0, 255, 180);",  # Púrpura
            "girar180": "background-color: rgba(192, 0, 255, 180);",  # Púrpura claro
            "sentarse": "background-color: rgba(0, 255, 128, 180);",  # Verde
            "ponerseDePie": "background-color: rgba(128, 255, 0, 180);",  # Verde amarillento
            "ninguna": "background-color: rgba(0, 0, 0, 180);"  # Negro
        }
        
        # Traducir actividad para mostrar en español
        actividad_es = {
            "caminarHacia": "CAMINANDO HACIA CÁMARA",
            "caminarRegreso": "CAMINANDO ALEJÁNDOSE",
            "girar90": "GIRANDO 90°",
            "girar180": "GIRANDO 180°",
            "sentarse": "SENTÁNDOSE",
            "ponerseDePie": "PONIÉNDOSE DE PIE",
            "ninguna": "NINGUNA"
        }.get(self.current_activity, self.current_activity.upper())
        
        # Estilo base
        base_style = """
            color: white; 
            font-size: 24px; 
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        """
        
        # Obtener color para la actividad actual
        color_style = activity_colors.get(activity, activity_colors["ninguna"])
        
        # Establecer estilo completo
        self.activity_label.setStyleSheet(color_style + base_style)
        self.activity_label.setText(f"PROBANDO: {actividad_es}")
        
        # Actualizar la métrica de actividad
        self.activity_metrics["actividad_actual"].setText(f"Actividad Actual: {self.current_activity} (Prueba)")
        
        print(f"PROBANDO ACTIVIDAD: {activity}")
        
        # Mostrar pistas visuales de movimiento
        self.video_thread.set_show_movement_hints(True, activity)
        
        # Establecer un temporizador para desactivar las pistas después de 10 segundos
        QTimer.singleShot(10000, lambda: self.video_thread.set_show_movement_hints(False))
        
        # Mostrar mensaje con las características esperadas para esta actividad
        expected_features = {
            "caminarHacia": "• Nariz: posición Z < -0.02\n• Movimiento: hacia la cámara",
            "caminarRegreso": "• Nariz: posición Z > 0.02\n• Movimiento: alejándose de la cámara",
            "girar90": "• Hombros: cambio de orientación > 25°\n• Pero menor a 60°",
            "girar180": "• Hombros: cambio de orientación > 60°",
            "sentarse": "• Rodillas: ángulo < 160°\n• Cadera: movimiento hacia abajo > 0.003",
            "ponerseDePie": "• Rodillas: ángulo < 160°\n• Cadera: movimiento hacia arriba < -0.003"
        }
        
        features = expected_features.get(activity, "No hay información disponible")
        QMessageBox.information(
            self,
            f"Probando {activity}",
            f"Características esperadas para '{activity}':\n\n{features}\n\n"
            f"Muévete realizando la actividad y observa si se detecta correctamente.\n"
            f"Se mostrarán pistas visuales durante 10 segundos."
        )
        
        # Añadir al historial visual con confianza simulada para pruebas
        self.history_widget.add_activity(activity, 90.0)
        
        # Actualizar barras de confianza para mostrar la actividad de prueba
        for act in self.activity_confidence_bars:
            value = 90 if act == activity else 0
            self.activity_confidence_bars[act].setValue(value)
        
        if "confianza_progress" in self.activity_metrics:
            self.activity_metrics["confianza_progress"].setValue(90)

    def show_thresholds_dialog(self):
        """Muestra un diálogo para ajustar los umbrales de detección"""
        dialog = ThresholdsDialog(self.video_thread, self)
        dialog.exec_()

    def _update_activity_confidence_bars(self, historial):
        """Actualiza las barras de confianza para cada actividad basada en el historial de predicciones"""
        if not historial:
            return
            
        # Calcular frecuencia de cada actividad en el historial
        activities = ["caminarHacia", "caminarRegreso", "girar90", "girar180", "sentarse", "ponerseDePie"]
        total_predictions = len(historial)
        
        for activity in activities:
            # Contar ocurrencias
            count = historial.count(activity)
            confidence = (count / total_predictions) * 100 if total_predictions > 0 else 0
            
            # Actualizar barra de progreso
            if activity in self.activity_confidence_bars:
                self.activity_confidence_bars[activity].setValue(int(confidence))

    def change_classifier(self):
        """Cambia entre el clasificador simple y el ML"""
        use_ml = self.classifier_selector.currentText() == "ML (Entrenado)"
        self.video_thread.switch_classifier(use_ml)
        print(f"Clasificador actual: {'ML' if use_ml else 'Simple'}")


# Función principal
def main(camera_id=0):
    app = QApplication(sys.argv)
    window = MainWindow()
    # Configurar el ID de la cámara seleccionada
    window.video_thread.video_capture.camera_id = camera_id
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 