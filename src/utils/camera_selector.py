import cv2
import os
import subprocess
import re
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QComboBox, QDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

def list_cameras_macos():
    """
    Lista las cámaras disponibles en macOS utilizando el comando system_profiler.
    
    Returns:
        list: Lista de tuplas (id, nombre_camara)
    """
    try:
        
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        
        cameras = []
        camera_blocks = result.stdout.split("Camera:")
        
        for i, block in enumerate(camera_blocks[1:], 0):  
            
            match = re.search(r"^\s*([^:]+)", block.strip())
            if match:
                camera_name = match.group(1).strip()
                cameras.append((i, camera_name))
        
        return cameras
    
    except subprocess.SubprocessError:
        print("Error al obtener la lista de cámaras con system_profiler")
        
        return fallback_camera_detection()

def fallback_camera_detection():
    """
    Método alternativo para detectar cámaras probando índices.
    
    Returns:
        list: Lista de tuplas (id, "Cámara {id}")
    """
    cameras = []
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cameras.append((i, f"Cámara {i}"))
            cap.release()
    
    return cameras

class CameraSelector(QDialog):
    """Diálogo para seleccionar una cámara"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar Cámara")
        self.setMinimumSize(640, 520)
        
        
        self.current_camera_id = 0
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        
        layout = QVBoxLayout(self)
        
        
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Seleccionar cámara:"))
        
        self.camera_combo = QComboBox()
        self.populate_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)
        camera_layout.addWidget(self.camera_combo)
        
        layout.addLayout(camera_layout)
        
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.preview_label)
        
        
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Actualizar lista")
        self.refresh_button.clicked.connect(self.refresh_camera_list)
        
        self.ok_button = QPushButton("Seleccionar")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        
        if self.camera_combo.count() > 0:
            self.on_camera_selected(0)
    
    def populate_camera_list(self):
        """Rellena el combobox con las cámaras disponibles"""
        self.camera_combo.clear()
        
        cameras = list_cameras_macos()
        if not cameras:
            self.camera_combo.addItem("No se encontraron cámaras", -1)
            return
        
        for camera_id, camera_name in cameras:
            self.camera_combo.addItem(f"{camera_name} (ID: {camera_id})", camera_id)
    
    def refresh_camera_list(self):
        """Actualiza la lista de cámaras"""
        if self.cap:
            self.stop_preview()
        
        self.populate_camera_list()
        
        if self.camera_combo.count() > 0:
            self.on_camera_selected(0)
    
    def on_camera_selected(self, index):
        """Maneja el cambio de cámara seleccionada"""
        
        if self.cap:
            self.stop_preview()
        
        
        camera_id = self.camera_combo.itemData(index)
        if camera_id == -1:
            return
        
        self.current_camera_id = camera_id
        
        
        self.start_preview()
    
    def start_preview(self):
        """Inicia la previsualización de la cámara"""
        self.cap = cv2.VideoCapture(self.current_camera_id)
        if self.cap.isOpened():
            
            self.timer.start(30)  
        else:
            self.preview_label.setText(f"Error al abrir la cámara {self.current_camera_id}")
    
    def stop_preview(self):
        """Detiene la previsualización de la cámara"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview_label.clear()
        self.preview_label.setStyleSheet("background-color: black;")
    
    def update_frame(self):
        """Actualiza el frame de previsualización"""
        if not self.cap or not self.cap.isOpened():
            self.stop_preview()
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_preview()
            return
        
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        
        self.preview_label.setPixmap(scaled_pixmap)
    
    def get_selected_camera_id(self):
        """Devuelve el ID de la cámara seleccionada"""
        return self.current_camera_id
    
    def closeEvent(self, event):
        """Maneja el evento de cierre de la ventana"""
        self.stop_preview()
        event.accept()


def select_camera():
    """
    Muestra un diálogo para seleccionar una cámara.
    
    Returns:
        int: ID de la cámara seleccionada o 0 si se cancela
    """
    app = QApplication.instance()
    if not app:
        app = QApplication([])
    
    selector = CameraSelector()
    result = selector.exec_()
    
    if result == QDialog.Accepted:
        return selector.get_selected_camera_id()
    else:
        return 0

if __name__ == "__main__":
    
    selected_camera = select_camera()
    print(f"Cámara seleccionada: ID {selected_camera}")
    
    
    if selected_camera >= 0:
        cap = cv2.VideoCapture(selected_camera)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Prueba Cámara {selected_camera}", frame)
                cv2.waitKey(0)
            cap.release()
        cv2.destroyAllWindows() 