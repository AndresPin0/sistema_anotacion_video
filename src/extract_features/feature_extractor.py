#!/usr/bin/env python3
"""
Feature Extractor Mejorado
=========================

Este módulo implementa la extracción de características biomecánicas mejoradas
incluyendo velocidad articular, ángulos articulares y deltas de movimiento.
"""

import numpy as np
from typing import Dict, List, Tuple
import mediapipe as mp

class FeatureExtractor:
    def __init__(self):
        """Inicializa el extractor de características."""
        self.prev_landmarks = None
        self.frame_count = 0
        self.temporal_window = 5  # Ventana temporal para cálculos
        self.landmark_history = []
        
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calcula el ángulo entre tres puntos en grados.
        
        Args:
            p1, p2, p3: Puntos 3D (x, y, z)
        Returns:
            float: Ángulo en grados
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_velocity(self, current: np.ndarray, previous: np.ndarray, fps: float = 30.0) -> float:
        """
        Calcula la velocidad entre dos posiciones.
        
        Args:
            current: Posición actual
            previous: Posición anterior
            fps: Cuadros por segundo
        Returns:
            float: Velocidad en unidades/segundo
        """
        if previous is None:
            return 0.0
        return np.linalg.norm(current - previous) * fps
    
    def _calculate_acceleration(self, velocities: List[float], fps: float = 30.0) -> float:
        """
        Calcula la aceleración basada en velocidades.
        
        Args:
            velocities: Lista de velocidades
            fps: Cuadros por segundo
        Returns:
            float: Aceleración en unidades/segundo²
        """
        if len(velocities) < 2:
            return 0.0
        return (velocities[-1] - velocities[-2]) * fps
    
    def extract_features(self, landmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extrae características biomecánicas mejoradas de los landmarks.
        
        Args:
            landmarks: Diccionario de landmarks con posiciones 3D
        Returns:
            Dict[str, float]: Características extraídas
        """
        features = {}
        
        # 1. Ángulos Articulares
        # Rodillas
        features['right_knee_angle'] = self._calculate_angle(
            landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle'])
        features['left_knee_angle'] = self._calculate_angle(
            landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle'])
        
        # Caderas
        features['right_hip_angle'] = self._calculate_angle(
            landmarks['right_shoulder'], landmarks['right_hip'], landmarks['right_knee'])
        features['left_hip_angle'] = self._calculate_angle(
            landmarks['left_shoulder'], landmarks['left_hip'], landmarks['left_knee'])
        
        # Tronco
        features['trunk_forward_tilt'] = self._calculate_angle(
            landmarks['nose'], 
            (landmarks['left_hip'] + landmarks['right_hip']) / 2,
            (landmarks['left_ankle'] + landmarks['right_ankle']) / 2
        )
        
        features['trunk_lateral_tilt'] = self._calculate_angle(
            landmarks['left_shoulder'],
            (landmarks['left_hip'] + landmarks['right_hip']) / 2,
            landmarks['right_shoulder']
        )
        
        # 2. Velocidades Articulares
        if self.prev_landmarks is not None:
            for joint in ['knee', 'hip', 'ankle', 'shoulder']:
                for side in ['left', 'right']:
                    key = f'{side}_{joint}'
                    features[f'{key}_velocity'] = self._calculate_velocity(
                        landmarks[key], 
                        self.prev_landmarks[key]
                    )
        
        # 3. Características de Movimiento Global
        # Centro de masa aproximado (COM)
        com = np.mean([landmarks['left_hip'], landmarks['right_hip']], axis=0)
        if self.prev_landmarks is not None:
            prev_com = np.mean([
                self.prev_landmarks['left_hip'], 
                self.prev_landmarks['right_hip']
            ], axis=0)
            
            features['com_velocity'] = self._calculate_velocity(com, prev_com)
            features['vertical_movement'] = com[1] - prev_com[1]
            features['forward_movement'] = com[2] - prev_com[2]
        
        # 4. Características de Simetría
        features['knee_angle_symmetry'] = abs(
            features['right_knee_angle'] - features['left_knee_angle'])
        features['hip_angle_symmetry'] = abs(
            features['right_hip_angle'] - features['left_hip_angle'])
        
        # 5. Características Temporales
        self.landmark_history.append(landmarks)
        if len(self.landmark_history) > self.temporal_window:
            self.landmark_history.pop(0)
            
            # Variación temporal de ángulos
            for joint in ['knee', 'hip']:
                for side in ['left', 'right']:
                    angles = [
                        self._calculate_angle(
                            hist[f'{side}_{joint.replace("knee", "hip")}'],
                            hist[f'{side}_{joint}'],
                            hist[f'{side}_{joint.replace("knee", "ankle")}']
                        )
                        for hist in self.landmark_history
                    ]
                    features[f'{side}_{joint}_angle_variance'] = np.var(angles)
        
        # 6. Visibilidad de Landmarks
        for key in landmarks:
            features[f'{key}_visibility'] = landmarks[key][3] if len(landmarks[key]) > 3 else 1.0
        
        # Actualizar estado
        self.prev_landmarks = landmarks.copy()
        self.frame_count += 1
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna los nombres de todas las características extraídas.
        
        Returns:
            List[str]: Lista de nombres de características
        """
        return [
            # Ángulos
            'right_knee_angle', 'left_knee_angle',
            'right_hip_angle', 'left_hip_angle',
            'trunk_forward_tilt', 'trunk_lateral_tilt',
            
            # Velocidades
            'right_knee_velocity', 'left_knee_velocity',
            'right_hip_velocity', 'left_hip_velocity',
            'right_ankle_velocity', 'left_ankle_velocity',
            'right_shoulder_velocity', 'left_shoulder_velocity',
            
            # Movimiento Global
            'com_velocity', 'vertical_movement', 'forward_movement',
            
            # Simetría
            'knee_angle_symmetry', 'hip_angle_symmetry',
            
            # Variaciones Temporales
            'right_knee_angle_variance', 'left_knee_angle_variance',
            'right_hip_angle_variance', 'left_hip_angle_variance',
            
            # Visibilidad
            'right_knee_visibility', 'left_knee_visibility',
            'right_hip_visibility', 'left_hip_visibility',
            'right_ankle_visibility', 'left_ankle_visibility',
            'right_shoulder_visibility', 'left_shoulder_visibility'
        ] 