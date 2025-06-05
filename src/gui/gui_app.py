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
        
        
        self.use_ml_classifier = False
        self.simple_classifier = SimpleActivityClassifier()
        self.ml_classifier = None
        self.activity_classifier = self.simple_classifier
        
        
        try:
            self.ml_classifier = MLActivityClassifier()
            print("✅ Clasificador ML cargado exitosamente")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el clasificador ML: {e}")
            print("   Usando clasificador simple por defecto")
        
        self.running = False
        
        
        self.activity_update_counter = 0
        self.activity_update_frequency = 10  
        
        
        self.last_reported_activity = None
        self.activity_confirmation_count = 0
        self.activity_confirmation_threshold = 3  
        self.no_activity_counter = 0  
        
        
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
                
                
                results_dict, processed_frame = self.pose_detector.process_frame(frame)
                
                
                if self.show_movement_hints:
                    processed_frame = self.add_movement_hints(
                        processed_frame, 
                        results_dict.get("landmarks", None)
                    )
                
                
                self.update_frame.emit(processed_frame)
                if results_dict and results_dict["landmarks"]:
                    self.update_data.emit(results_dict)
                    
                    
                    current_activity = self.activity_classifier.predict(
                        results_dict["landmarks"], 
                        results_dict.get("angles", {})
                    )
                    
                    
                    if current_activity == self.last_reported_activity:
                        
                        self.activity_confirmation_count += 1
                    else:
                        
                        self.activity_confirmation_count = 1
                        
                    
                    
                    
                    should_update = (
                        self.activity_confirmation_count >= self.activity_confirmation_threshold or
                        (self.activity_update_counter % self.activity_update_frequency == 0 and 
                         self.activity_confirmation_count >= 1)
                    )
                    
                    if should_update:
                        self.update_activity.emit(current_activity)
                        self.last_reported_activity = current_activity
                        print(f"🎯 Actividad reportada: {current_activity} (confirmaciones: {self.activity_confirmation_count})")
                        self.no_activity_counter = 0  
                    else:
                        
                        if self.activity_update_counter % 30 == 0:  
                            print(f"🔍 Detectando: {current_activity} (confirmaciones: {self.activity_confirmation_count}/{self.activity_confirmation_threshold})")
                        
                        
                        if current_activity in ["ninguna", "desconocida"]:
                            self.no_activity_counter += 1
                            
                            if self.no_activity_counter > 60:  
                                self.update_activity.emit("ninguna")
                                self.last_reported_activity = "ninguna"
                                self.no_activity_counter = 0
                                print("⭕ Sin actividad detectada - mostrando 'ninguna'")
                    
                    self.activity_update_counter += 1
                
                
                time.sleep(0.01)
                
            
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
        
        
        overlay = frame.copy()
        alpha = 0.6  
        
        
        activity_color = {
            "caminarHacia": (0, 128, 255),  
            "caminarRegreso": (0, 192, 255),  
            "girar90": (128, 0, 255),  
            "girar180": (192, 0, 255),  
            "sentarse": (0, 255, 128),  
            "ponerseDePie": (128, 255, 0)   
        }
        
        color = activity_color.get(activity, (200, 200, 200))
        
        
        if activity in ["caminarHacia", "caminarRegreso"]:
            
            arrow_start = (w // 2, h // 2)
            if activity == "caminarHacia":
                
                arrow_end = (w // 2, h // 2 + 100)
                text = "Camina HACIA la cámara"
            else:
                
                arrow_end = (w // 2, h // 2 - 100)
                text = "Camina ALEJÁNDOTE de la cámara"
                
            
            cv2.arrowedLine(overlay, arrow_start, arrow_end, color, 5, tipLength=0.3)
            
        elif activity in ["girar90", "girar180"]:
            
            center = (w // 2, h // 2)
            radius = 100
            angle = 90 if activity == "girar90" else 180
            text = f"Gira {angle}°"
            
            
            cv2.circle(overlay, center, radius, color, 2)
            
            
            start_angle = 0
            end_angle = np.radians(angle)
            
            
            start_point = (int(center[0] + radius), center[1])
            end_x = int(center[0] + radius * np.cos(end_angle))
            end_y = int(center[1] - radius * np.sin(end_angle))
            end_point = (end_x, end_y)
            
            
            cv2.line(overlay, center, start_point, color, 2)
            cv2.line(overlay, center, end_point, color, 2)
            
            
            cv2.ellipse(overlay, center, (radius, radius), 0, 0, angle, color, 3)
            
        elif activity in ["sentarse", "ponerseDePie"]:
            
            if activity == "sentarse":
                
                arrow_start = (w // 2, h // 3)
                arrow_end = (w // 2, 2 * h // 3)
                text = "Siéntate en una silla"
            else:
                
                arrow_start = (w // 2, 2 * h // 3)
                arrow_end = (w // 2, h // 3)
                text = "Levántate de la silla"
                
            
            cv2.arrowedLine(overlay, arrow_start, arrow_end, color, 5, tipLength=0.3)
            
            
            if landmarks:
                
                pass
        
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        
        cv2.putText(frame, text, (w // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame


class ThresholdsDialog(QDialog):
    """Diálogo para ajustar los umbrales de detección de actividades en tiempo real"""
    
    def __init__(self, video_thread, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajustar Umbrales de Detección")
        self.setMinimumSize(500, 400)
        self.video_thread = video_thread
        
        
        self.current_thresholds = self.video_thread.get_thresholds()
        
        
        self._create_ui()
    
    def _create_ui(self):
        """Crea la interfaz del diálogo de umbrales"""
        layout = QVBoxLayout(self)
        
        
        walking_group = QGroupBox("Umbrales para Caminar")
        walking_layout = QFormLayout(walking_group)
        
        
        self.z_threshold_spin = QDoubleSpinBox()
        self.z_threshold_spin.setRange(0.001, 0.1)
        self.z_threshold_spin.setSingleStep(0.001)
        self.z_threshold_spin.setDecimals(3)
        self.z_threshold_spin.setValue(self.current_thresholds["z_threshold"])
        self.z_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("z_threshold", value)
        )
        walking_layout.addRow("Umbral Z (caminar):", self.z_threshold_spin)
        
        
        walking_layout.addRow(QLabel("Valores más bajos = más sensible a detectar movimiento hacia/desde cámara"))
        
        layout.addWidget(walking_group)
        
        
        turn_group = QGroupBox("Umbrales para Giros")
        turn_layout = QFormLayout(turn_group)
        
        
        self.orientation_threshold_spin = QDoubleSpinBox()
        self.orientation_threshold_spin.setRange(1.0, 30.0)
        self.orientation_threshold_spin.setSingleStep(0.5)
        self.orientation_threshold_spin.setDecimals(1)
        self.orientation_threshold_spin.setValue(self.current_thresholds["orientation_threshold"])
        self.orientation_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("orientation_threshold", value)
        )
        turn_layout.addRow("Umbral Orientación (giro 90°):", self.orientation_threshold_spin)
        
        
        self.orientation_threshold_large_spin = QDoubleSpinBox()
        self.orientation_threshold_large_spin.setRange(20.0, 90.0)
        self.orientation_threshold_large_spin.setSingleStep(1.0)
        self.orientation_threshold_large_spin.setDecimals(1)
        self.orientation_threshold_large_spin.setValue(self.current_thresholds["orientation_threshold_large"])
        self.orientation_threshold_large_spin.valueChanged.connect(
            lambda value: self.update_threshold("orientation_threshold_large", value)
        )
        turn_layout.addRow("Umbral Orientación Grande (giro 180°):", self.orientation_threshold_large_spin)
        
        
        turn_layout.addRow(QLabel("Valores más bajos = más sensible a detectar giros"))
        
        layout.addWidget(turn_group)
        
        
        sit_group = QGroupBox("Umbrales para Sentarse/Levantarse")
        sit_layout = QFormLayout(sit_group)
        
        
        self.knee_angle_threshold_spin = QDoubleSpinBox()
        self.knee_angle_threshold_spin.setRange(100.0, 180.0)
        self.knee_angle_threshold_spin.setSingleStep(1.0)
        self.knee_angle_threshold_spin.setDecimals(1)
        self.knee_angle_threshold_spin.setValue(self.current_thresholds["knee_angle_threshold"])
        self.knee_angle_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("knee_angle_threshold", value)
        )
        sit_layout.addRow("Umbral Ángulo Rodilla:", self.knee_angle_threshold_spin)
        
        
        self.hip_movement_threshold_spin = QDoubleSpinBox()
        self.hip_movement_threshold_spin.setRange(0.0001, 0.01)
        self.hip_movement_threshold_spin.setSingleStep(0.0001)
        self.hip_movement_threshold_spin.setDecimals(4)
        self.hip_movement_threshold_spin.setValue(self.current_thresholds["hip_movement_threshold"])
        self.hip_movement_threshold_spin.valueChanged.connect(
            lambda value: self.update_threshold("hip_movement_threshold", value)
        )
        sit_layout.addRow("Umbral Movimiento Cadera:", self.hip_movement_threshold_spin)
        
        
        sit_layout.addRow(QLabel("Valores más altos en ángulo = más sensible a flexión de rodilla"))
        sit_layout.addRow(QLabel("Valores más bajos en movimiento = más sensible a movimiento de cadera"))
        
        layout.addWidget(sit_group)
        
        
        button_layout = QHBoxLayout()
        
        
        defaults_button = QPushButton("Restaurar Valores Predeterminados")
        defaults_button.clicked.connect(self.restore_defaults)
        button_layout.addWidget(defaults_button)
        
        
        button_layout.addStretch()
        
        
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
        
        
        self.z_threshold_spin.setValue(default_thresholds["z_threshold"])
        self.knee_angle_threshold_spin.setValue(default_thresholds["knee_angle_threshold"])
        self.hip_movement_threshold_spin.setValue(default_thresholds["hip_movement_threshold"])
        self.orientation_threshold_spin.setValue(default_thresholds["orientation_threshold"])
        self.orientation_threshold_large_spin.setValue(default_thresholds["orientation_threshold_large"])
        
        
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
        
        
        self.activity_history = []
        self.max_history_length = 20
        
        
        self.activity_colors = {
            "caminarHacia": QColor(0, 128, 255),  
            "caminarRegreso": QColor(0, 192, 255),  
            "girar90": QColor(128, 0, 255),  
            "girar180": QColor(192, 0, 255),  
            "sentarse": QColor(0, 255, 128),  
            "ponerseDePie": QColor(128, 255, 0),  
            "desconocida": QColor(128, 128, 128),  
            "ninguna": QColor(64, 64, 64)  
        }
        
        
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
        
        
        painter.fillRect(event.rect(), QBrush(QColor(30, 30, 30)))
        
        
        if not self.activity_history:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(event.rect(), Qt.AlignCenter, "No hay actividades registradas")
            return
        
        
        width = self.width()
        height = self.height()
        block_width = width / min(self.max_history_length, max(1, len(self.activity_history)))
        
        
        x = 0
        for i, (activity, confidence) in enumerate(self.activity_history):
            
            color = self.activity_colors.get(activity, QColor(100, 100, 100))
            
            
            block_height = height * min(1.0, max(0.2, confidence / 100))
            y = height - block_height
            
            
            painter.fillRect(QRect(int(x), int(y), int(block_width), int(block_height)), QBrush(color))
            
            
            painter.setPen(Qt.white)
            activity_text = self.activity_names.get(activity, activity)
            text_rect = QRect(int(x), int(y), int(block_width), int(block_height))
            painter.drawText(text_rect, Qt.AlignCenter, activity_text)
            
            
            painter.setPen(QPen(QColor(50, 50, 50), 1))
            painter.drawLine(int(x + block_width), 0, int(x + block_width), height)
            
            
            x += block_width
        
        
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(0, 0, width - 1, height - 1)
        
        
        painter.setPen(Qt.white)
        painter.drawText(QRect(5, height - 20, 50, 15), Qt.AlignLeft, "Anterior")
        painter.drawText(QRect(width - 55, height - 20, 50, 15), Qt.AlignRight, "Actual")


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Sistema de Anotación de Video - Análisis de Actividades")
        self.setMinimumSize(1280, 800)
        
        
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: 
                color: 
            }
            QGroupBox {
                border: 1px solid 
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: 
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QPushButton {
                background-color: 
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: 
            }
            QPushButton:pressed {
                background-color: 
            }
            QPushButton:disabled {
                background-color: 
            }
            QLabel {
                color: 
            }
            QTextEdit {
                background-color: 
                color: 
                border: 1px solid 
                border-radius: 3px;
            }
        """)
        
        
        self.video_thread = VideoThread()
        self.video_thread.update_frame.connect(self.update_video_frame)
        self.video_thread.update_data.connect(self.update_data_display)
        self.video_thread.update_activity.connect(self.update_activity_display)
        self.video_thread.error_occurred.connect(self.show_error)
        
        self.recording = False
        self.current_activity = "ninguna"
        self.auto_detect_activity = True  
        
        
        self._create_ui()
        
        
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
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)  
        
        
        video_panel = QVBoxLayout()
        video_panel.setSpacing(8)  
        
        
        video_title = QLabel("Vista de Cámara y Detección de Actividades")
        video_title.setAlignment(Qt.AlignCenter)
        video_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px; color: 
        video_panel.addWidget(video_title)
        
        
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container.setStyleSheet("background-color: black; border-radius: 5px;")
        
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: transparent;")
        video_container_layout.addWidget(self.video_label)
        
        
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
        
        
        history_title = QLabel("Historial de Actividades")
        history_title.setAlignment(Qt.AlignCenter)
        history_title.setStyleSheet("font-size: 14px; margin-top: 5px; color: 
        video_panel.addWidget(history_title)
        
        
        self.history_widget = ActivityHistoryWidget()
        video_panel.addWidget(self.history_widget)
        
        
        controls_group = QGroupBox("Controles de Actividad")
        controls_layout = QVBoxLayout(controls_group)
        
        
        test_panel = QHBoxLayout()
        test_panel.addWidget(QLabel("Prueba de Actividades:"))
        
        
        button_layout = QHBoxLayout()
        self.test_buttons = {}
        
        
        walking_layout = QHBoxLayout()
        walking_title = QLabel("Caminar:")
        walking_title.setFixedWidth(80)
        walking_layout.addWidget(walking_title)
        
        for activity in ["caminarHacia", "caminarRegreso"]:
            btn = QPushButton(activity.capitalize().replace("caminar", ""))
            btn.setFixedHeight(35)
            if activity == "caminarHacia":
                btn.setStyleSheet("background-color: 
            else:
                btn.setStyleSheet("background-color: 
            btn.clicked.connect(lambda checked, act=activity: self.test_activity(act))
            walking_layout.addWidget(btn)
            self.test_buttons[activity] = btn
        
        controls_layout.addLayout(walking_layout)
        
        
        turn_layout = QHBoxLayout()
        turn_title = QLabel("Girar:")
        turn_title.setFixedWidth(80)
        turn_layout.addWidget(turn_title)
        
        for activity in ["girar90", "girar180"]:
            btn = QPushButton(activity.capitalize().replace("girar", ""))
            btn.setFixedHeight(35)
            if activity == "girar90":
                btn.setStyleSheet("background-color: 
            else:
                btn.setStyleSheet("background-color: 
            btn.clicked.connect(lambda checked, act=activity: self.test_activity(act))
            turn_layout.addWidget(btn)
            self.test_buttons[activity] = btn
        
        controls_layout.addLayout(turn_layout)
        
        
        sit_layout = QHBoxLayout()
        sit_title = QLabel("Postura:")
        sit_title.setFixedWidth(80)
        sit_layout.addWidget(sit_title)
        
        for activity in ["sentarse", "ponerseDePie"]:
            btn = QPushButton(activity.capitalize())
            btn.setFixedHeight(35)
            if activity == "sentarse":
                btn.setStyleSheet("background-color: 
            else:
                btn.setStyleSheet("background-color: 
            btn.clicked.connect(lambda checked, act=activity: self.test_activity(act))
            sit_layout.addWidget(btn)
            self.test_buttons[activity] = btn
        
        controls_layout.addLayout(sit_layout)
        
        video_panel.addWidget(controls_group)
        
        
        camera_controls = QVBoxLayout()
        
        
        first_row = QHBoxLayout()
        
        
        self.camera_button = QPushButton("Iniciar Cámara")
        self.camera_button.setFixedHeight(40)
        self.camera_button.setStyleSheet("background-color: 
        self.camera_button.clicked.connect(self.toggle_camera)
        first_row.addWidget(self.camera_button)
        
        
        self.auto_detect_button = QPushButton("Detección Auto: ON")
        self.auto_detect_button.setFixedHeight(40)
        self.auto_detect_button.setCheckable(True)
        self.auto_detect_button.setChecked(True)
        self.auto_detect_button.setStyleSheet("""
            QPushButton:checked { background-color: 
            QPushButton:!checked { background-color: 
        """)
        self.auto_detect_button.clicked.connect(self.toggle_auto_detection)
        first_row.addWidget(self.auto_detect_button)
        
        camera_controls.addLayout(first_row)
        
        
        second_row = QHBoxLayout()
        
        
        classifier_container = QHBoxLayout()
        classifier_label = QLabel("Clasificador:")
        classifier_container.addWidget(classifier_label)
        
        self.classifier_selector = QComboBox()
        self.classifier_selector.addItems(["Simple (Reglas)", "ML (Entrenado)"])
        self.classifier_selector.setFixedHeight(40)
        self.classifier_selector.currentTextChanged.connect(self.change_classifier)
        classifier_container.addWidget(self.classifier_selector)
        
        second_row.addLayout(classifier_container)
        
        
        self.adjust_thresholds_button = QPushButton("Ajustar Umbrales")
        self.adjust_thresholds_button.setFixedHeight(40)
        self.adjust_thresholds_button.setStyleSheet("background-color: 
        self.adjust_thresholds_button.clicked.connect(self.show_thresholds_dialog)
        second_row.addWidget(self.adjust_thresholds_button)
        
        camera_controls.addLayout(second_row)
        
        video_panel.addLayout(camera_controls)
        
        
        data_panel = QVBoxLayout()
        data_panel.setSpacing(10)  
        
        
        data_title = QLabel("Datos y Métricas")
        data_title.setAlignment(Qt.AlignCenter)
        data_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px; color: 
        data_panel.addWidget(data_title)
        
        
        activity_group = QGroupBox("Actividad Actual y Métricas")
        activity_layout = QVBoxLayout(activity_group)
        
        
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
                
                confidence_layout = QHBoxLayout()
                
                
                label = QLabel(text)
                label.setStyleSheet("font-size: 14px;")
                confidence_layout.addWidget(label, 2)  
                
                
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                progress_bar.setValue(0)
                progress_bar.setFormat("%v%")
                progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid 
                        border-radius: 3px;
                        text-align: center;
                        background-color: 
                        height: 20px;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 
                        border-radius: 2px;
                    }
                """)
                confidence_layout.addWidget(progress_bar, 3)
                
                
                self.activity_metrics[key] = label
                self.activity_metrics[key + "_progress"] = progress_bar
                
                
                activity_layout.addLayout(confidence_layout)
            else:
                
                label = QLabel(text)
                label.setStyleSheet("font-size: 14px;")
                activity_layout.addWidget(label)
                self.activity_metrics[key] = label
        
        
        confidence_group = QGroupBox("Confianza por Actividad")
        confidence_layout = QVBoxLayout(confidence_group)
        
        
        self.activity_confidence_bars = {}
        for activity in ["caminarHacia", "caminarRegreso", "girar90", "girar180", "sentarse", "ponerseDePie"]:
            
            activity_bar_layout = QHBoxLayout()
            
            
            label = QLabel(activity.capitalize())
            label.setFixedWidth(120)
            activity_bar_layout.addWidget(label)
            
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("%v%")
            
            
            if activity in ["caminarHacia", "caminarRegreso"]:
                color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 
            elif activity in ["girar90", "girar180"]:
                color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 
            else:  
                color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 
            
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid 
                    border-radius: 3px;
                    text-align: center;
                    background-color: 
                    height: 15px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 2px;
                }}
            """)
            
            activity_bar_layout.addWidget(progress_bar)
            
            
            self.activity_confidence_bars[activity] = progress_bar
            
            
            confidence_layout.addLayout(activity_bar_layout)
        
        data_panel.addWidget(activity_group)
        data_panel.addWidget(confidence_group)
        
        
        angles_group = QGroupBox("Ángulos Articulares")
        angles_layout = QVBoxLayout(angles_group)
        
        
        self.angle_labels = {}
        for angle_name in ["left_knee_angle", "right_knee_angle", "left_hip_angle", 
                          "right_hip_angle", "left_elbow_angle", "right_elbow_angle", 
                          "trunk_lateral_inclination"]:
            label = QLabel(f"{angle_name}: --")
            angles_layout.addWidget(label)
            self.angle_labels[angle_name] = label
        
        data_panel.addWidget(angles_group)
        
        
        landmarks_group = QGroupBox("Landmarks Detectados")
        landmarks_layout = QVBoxLayout(landmarks_group)
        
        self.landmarks_text = QTextEdit()
        self.landmarks_text.setReadOnly(True)
        landmarks_layout.addWidget(self.landmarks_text)
        
        data_panel.addWidget(landmarks_group)
        
        
        main_layout.addLayout(video_panel, 2)
        main_layout.addLayout(data_panel, 1)
    
    def update_video_frame(self, frame):
        """Actualiza el frame de video en la interfaz"""
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        
        pixmap = QPixmap.fromImage(qt_image)
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_data_display(self, results_dict):
        """Actualiza la visualización de datos"""
        if not results_dict:
            return
        
        
        angles = results_dict.get("angles", {})
        if angles:
            for angle_name, angle_value in angles.items():
                if angle_name in self.angle_labels:
                    self.angle_labels[angle_name].setText(f"{angle_name}: {angle_value:.1f}°")
            
            
            if "left_knee_angle" in angles and "right_knee_angle" in angles:
                avg_knee = (angles["left_knee_angle"] + angles["right_knee_angle"]) / 2
                rodillas_estado = "Dobladas" if avg_knee < 150 else "Extendidas"
                self.activity_metrics["angulo_rodillas"].setText(f"Ángulo Rodillas: {avg_knee:.1f}° ({rodillas_estado})")
            
            if "trunk_lateral_inclination" in angles:
                inclinacion = angles["trunk_lateral_inclination"]
                lado = "Izquierda" if inclinacion > 5 else "Derecha" if inclinacion < -5 else "Centro"
                self.activity_metrics["inclinacion_tronco"].setText(f"Inclinación Tronco: {inclinacion:.1f}° ({lado})")
        
        
        landmarks = results_dict.get("landmarks", {})
        if landmarks:
            
            summary = "Landmarks detectados:\n"
            
            
            if "nose" in landmarks:
                
                nose_z = landmarks["nose"]["z"]
                estado = "Hacia la cámara" if nose_z < -0.03 else "Alejándose" if nose_z > 0.03 else "Estático"
                self.activity_metrics["estado_movimiento"].setText(f"Estado de Movimiento: {estado} (Z: {nose_z:.3f})")
            
            
            for landmark_name in ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
                if landmark_name in landmarks:
                    lm = landmarks[landmark_name]
                    summary += f"{landmark_name}: ({lm['x']:.2f}, {lm['y']:.2f}, {lm['z']:.2f})\n"
            
            self.landmarks_text.setText(summary)
    
    def update_activity_display(self, activity):
        """Actualiza la visualización de la actividad detectada"""
        
        if self.auto_detect_activity:
            
            if activity != "desconocida":
                
                if self.current_activity != activity:
                    self.current_activity = activity
                    
                    
                    activity_colors = {
                        "caminarHacia": "background-color: rgba(0, 128, 255, 180);",  
                        "caminarRegreso": "background-color: rgba(0, 192, 255, 180);",  
                        "girar90": "background-color: rgba(128, 0, 255, 180);",  
                        "girar180": "background-color: rgba(192, 0, 255, 180);",  
                        "sentarse": "background-color: rgba(0, 255, 128, 180);",  
                        "ponerseDePie": "background-color: rgba(128, 255, 0, 180);",  
                        "ninguna": "background-color: rgba(0, 0, 0, 180);"  
                    }
                    
                    
                    self.activity_metrics["actividad_actual"].setText(f"Actividad Actual: {self.current_activity}")
                    
                    
                    actividad_es = {
                        "caminarHacia": "CAMINANDO HACIA CÁMARA",
                        "caminarRegreso": "CAMINANDO ALEJÁNDOSE",
                        "girar90": "GIRANDO 90°",
                        "girar180": "GIRANDO 180°",
                        "sentarse": "SENTÁNDOSE",
                        "ponerseDePie": "PONIÉNDOSE DE PIE",
                        "ninguna": "NINGUNA"
                    }.get(self.current_activity, self.current_activity.upper())
                    
                    
                    base_style = """
                        color: white; 
                        font-size: 24px; 
                        font-weight: bold;
                        border-radius: 10px;
                        padding: 10px;
                    """
                    
                    
                    color_style = activity_colors.get(activity, activity_colors["ninguna"])
                    
                    
                    self.activity_label.setStyleSheet(color_style + base_style)
                    self.activity_label.setText(f"ACTIVIDAD: {actividad_es}")
                    
                    
                    historial = self.video_thread.activity_classifier.prediction_history if hasattr(self.video_thread.activity_classifier, 'prediction_history') else []
                    confianza = 0
                    if historial and len(historial) > 1:
                        actividad_actual_count = historial.count(self.current_activity)
                        confianza = actividad_actual_count / len(historial) * 100
                        nivel = "Alta" if confianza > 80 else "Media" if confianza > 50 else "Baja"
                        self.activity_metrics["confianza"].setText(f"Confianza: {confianza:.0f}% ({nivel})")
                        
                        
                        if "confianza_progress" in self.activity_metrics:
                            self.activity_metrics["confianza_progress"].setValue(int(confianza))
                    else:
                        
                        confirmation_ratio = self.video_thread.activity_confirmation_count / max(1, self.video_thread.activity_confirmation_threshold)
                        confianza = min(100, confirmation_ratio * 100)
                        nivel = "Alta" if confianza > 80 else "Media" if confianza > 50 else "Baja"
                        self.activity_metrics["confianza"].setText(f"Confianza: {confianza:.0f}% ({nivel})")
                        
                        if "confianza_progress" in self.activity_metrics:
                            self.activity_metrics["confianza_progress"].setValue(int(confianza))
                    
                    
                    self.history_widget.add_activity(self.current_activity, confianza)
                    
                    
                    self._update_activity_confidence_bars(historial)
                    
                    
                    print(f"ACTIVIDAD DETECTADA: {self.current_activity}")
                else:
                    
                    confirmation_ratio = self.video_thread.activity_confirmation_count / max(1, self.video_thread.activity_confirmation_threshold)
                    confianza = min(100, confirmation_ratio * 100)
                    nivel = "Alta" if confianza > 80 else "Media" if confianza > 50 else "Baja"
                    self.activity_metrics["confianza"].setText(f"Confianza: {confianza:.0f}% ({nivel})")
                    
                    if "confianza_progress" in self.activity_metrics:
                        self.activity_metrics["confianza_progress"].setValue(int(confianza))
    
    def toggle_camera(self):
        """Inicia o detiene la cámara"""
        if not self.video_thread.running:
            
            self.video_thread.start()
            self.camera_button.setText("Detener Cámara")
        else:
            
            self.video_thread.stop()
            self.camera_button.setText("Iniciar Cámara")
            
            
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: black;")
            
            
            for label in self.angle_labels.values():
                label.setText(label.text().split(":")[0] + ": --")
            
            self.landmarks_text.clear()
            
            
            self.video_thread.set_show_movement_hints(False)
    
    def toggle_auto_detection(self):
        """Habilita o deshabilita la detección automática de actividades"""
        self.auto_detect_activity = self.auto_detect_button.isChecked()
        button_text = "Detección Auto: ON" if self.auto_detect_activity else "Detección Auto: OFF"
        self.auto_detect_button.setText(button_text)
        print(f"Detección automática de actividad: {'ACTIVADA' if self.auto_detect_activity else 'DESACTIVADA'}")
        
        
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
        
        else:
            self.current_activity = "ninguna"
            self.activity_label.setText(f"ACTIVIDAD: {self.current_activity}")
            
            
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
        
        if self.video_thread.running:
            self.video_thread.stop()
        event.accept()

    def test_activity(self, activity):
        """Prueba una actividad específica actualizando la interfaz"""
        self.current_activity = activity
        
        
        activity_colors = {
            "caminarHacia": "background-color: rgba(0, 128, 255, 180);",  
            "caminarRegreso": "background-color: rgba(0, 192, 255, 180);",  
            "girar90": "background-color: rgba(128, 0, 255, 180);",  
            "girar180": "background-color: rgba(192, 0, 255, 180);",  
            "sentarse": "background-color: rgba(0, 255, 128, 180);",  
            "ponerseDePie": "background-color: rgba(128, 255, 0, 180);",  
            "ninguna": "background-color: rgba(0, 0, 0, 180);"  
        }
        
        
        actividad_es = {
            "caminarHacia": "CAMINANDO HACIA CÁMARA",
            "caminarRegreso": "CAMINANDO ALEJÁNDOSE",
            "girar90": "GIRANDO 90°",
            "girar180": "GIRANDO 180°",
            "sentarse": "SENTÁNDOSE",
            "ponerseDePie": "PONIÉNDOSE DE PIE",
            "ninguna": "NINGUNA"
        }.get(self.current_activity, self.current_activity.upper())
        
        
        base_style = """
            color: white; 
            font-size: 24px; 
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        """
        
        
        color_style = activity_colors.get(activity, activity_colors["ninguna"])
        
        
        self.activity_label.setStyleSheet(color_style + base_style)
        self.activity_label.setText(f"PROBANDO: {actividad_es}")
        
        
        self.activity_metrics["actividad_actual"].setText(f"Actividad Actual: {self.current_activity} (Prueba)")
        
        print(f"PROBANDO ACTIVIDAD: {activity}")
        
        
        self.video_thread.set_show_movement_hints(True, activity)
        
        
        QTimer.singleShot(10000, lambda: self.video_thread.set_show_movement_hints(False))
        
        
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
        
        
        self.history_widget.add_activity(activity, 90.0)
        
        
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
            
        
        activities = ["caminarHacia", "caminarRegreso", "girar90", "girar180", "sentarse", "ponerseDePie"]
        total_predictions = len(historial)
        
        for activity in activities:
            
            count = historial.count(activity)
            confidence = (count / total_predictions) * 100 if total_predictions > 0 else 0
            
            
            if activity in self.activity_confidence_bars:
                self.activity_confidence_bars[activity].setValue(int(confidence))

    def change_classifier(self):
        """Cambia entre el clasificador simple y el ML"""
        use_ml = self.classifier_selector.currentText() == "ML (Entrenado)"
        self.video_thread.switch_classifier(use_ml)
        print(f"Clasificador actual: {'ML' if use_ml else 'Simple'}")



def main(camera_id=0):
    app = QApplication(sys.argv)
    window = MainWindow()
    
    window.video_thread.video_capture.camera_id = camera_id
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 