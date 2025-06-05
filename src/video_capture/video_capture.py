import cv2
import os
import time
import numpy as np
from datetime import datetime

class VideoCapture:
    def __init__(self, camera_id=0):
        """
        Inicializa el módulo de captura de video.
        
        Args:
            camera_id: ID de la cámara a utilizar (0 por defecto - webcam principal)
        """
        self.camera_id = camera_id
        self.cap = None
    
    def start_camera(self):
        """Inicia la cámara"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"No se pudo abrir la cámara {self.camera_id}")
            
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap.isOpened()
    
    def stop_camera(self):
        """Detiene la cámara"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        """
        Captura un frame de la cámara.
        
        Returns:
            tuple: (success, frame, timestamp) o (False, None, None) si hay un error
        """
        if not self.cap or not self.cap.isOpened():
            return False, None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        timestamp = time.time()
        return ret, frame, timestamp



if __name__ == "__main__":
    
    video_capture = VideoCapture()
    
    try:
        
        video_capture.start_camera()
        
        print("Presiona 'q' para salir")
        
        while True:
            
            ret, frame, timestamp = video_capture.get_frame()
            if not ret:
                break
            
            
            cv2.imshow('Video Capture', frame)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        
        video_capture.stop_camera()
        cv2.destroyAllWindows() 