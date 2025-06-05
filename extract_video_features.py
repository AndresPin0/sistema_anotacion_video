 
"""
Extractor de Características de Video
==================================

Este script procesa videos de entrenamiento y extrae características
usando MediaPipe y nuestro extractor de características mejorado.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd

from src.extract_features.feature_extractor import FeatureExtractor

class VideoProcessor:
    def __init__(self):
        """Inicializa el procesador de video con MediaPipe."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2  
        )
        self.feature_extractor = FeatureExtractor()
        
    def _extract_landmarks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extrae landmarks de un frame usando MediaPipe.
        
        Args:
            frame: Frame del video en formato BGR
        Returns:
            Dict[str, np.ndarray]: Diccionario de landmarks
        """
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(idx).name.lower()
            landmarks[name] = np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
        return landmarks
        
    def process_video(self, video_path: str) -> List[Dict[str, float]]:
        """
        Procesa un video y extrae características de cada frame.
        
        Args:
            video_path: Ruta al archivo de video
        Returns:
            List[Dict[str, float]]: Lista de características por frame
        """
        cap = cv2.VideoCapture(video_path)
        features_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            
            landmarks = self._extract_landmarks(frame)
            if landmarks is not None:
                
                features = self.feature_extractor.extract_features(landmarks)
                features_list.append(features)
                
        cap.release()
        return features_list

def process_dataset(data_dir: str = "data/raw", 
                   output_dir: str = "data/processed") -> None:
    """
    Procesa todo el dataset de videos y guarda las características.
    
    Args:
        data_dir: Directorio con los videos y anotaciones
        output_dir: Directorio para guardar las características procesadas
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    annotations_file = Path(data_dir) / "annotations.json"
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    
    processor = VideoProcessor()
    
    
    all_features = []
    all_labels = []
    
    
    print("\nProcesando videos...")
    for video_info in tqdm(annotations):
        video_path = str(Path(data_dir) / video_info["filename"])
        label = video_info["activity"]
        
        try:
            
            video_features = processor.process_video(video_path)
            
            if video_features:
                
                avg_features = {}
                for feature_name in video_features[0].keys():
                    avg_features[feature_name] = np.mean([
                        frame[feature_name] for frame in video_features
                    ])
                
                all_features.append(list(avg_features.values()))
                all_labels.append(label)
                
        except Exception as e:
            print(f"\n⚠️  Error procesando {video_path}: {e}")
            continue
    
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    
    np.save(output_path / "features.npy", X)
    np.save(output_path / "labels.npy", y)
    
    
    feature_names = processor.feature_extractor.get_feature_names()
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)
    
    
    print("\nEstadísticas del dataset procesado:")
    print(f"- Total de muestras: {len(all_features)}")
    print(f"- Características por muestra: {len(feature_names)}")
    
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\nDistribución de clases:")
    for label, count in zip(unique_labels, counts):
        print(f"- {label}: {count} muestras")

def main():
    """Función principal."""
    print("=" * 60)
    print("EXTRACCIÓN DE CARACTERÍSTICAS")
    print("=" * 60)
    
    print("\nVerificando directorios...")
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("\n❌ Error: No se encuentra el directorio data/raw")
        print("Crea el directorio y coloca los videos y annotations.json")
        return
        
    if not (data_dir / "annotations.json").exists():
        print("\n❌ Error: No se encuentra annotations.json")
        print("Coloca el archivo de anotaciones en data/raw/")
        return
    
    print("✓ Directorios verificados")
    process_dataset()
    
    print("\n✅ Procesamiento completado!")
    print("\nArchivos generados:")
    print("- data/processed/features.npy")
    print("- data/processed/labels.npy")
    print("- data/processed/feature_names.json")

if __name__ == "__main__":
    main()