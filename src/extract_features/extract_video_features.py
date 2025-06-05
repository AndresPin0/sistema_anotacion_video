
"""
Extractor de Características de Video
====================================

Este script procesa los videos en las carpetas de actividades y extrae
características relevantes usando MediaPipe para entrenar modelos de clasificación.

Funcionalidades:
- Procesa videos por actividad
- Extrae landmarks y ángulos articulares
- Calcula características temporales (velocidades, aceleraciones)
- Guarda datos en formato CSV para entrenamiento
- Genera estadísticas del dataset

Uso:
    python extract_video_features.py
    python extract_video_features.py --output_dir data/training_data --video_dir videos
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.pose_detection.pose_detector import PoseDetector
from src.video_capture.video_capture import VideoCapture

class VideoFeatureExtractor:
    """
    Extractor de características de videos para entrenamiento de modelos.
    """
    
    def __init__(self, output_dir="data/training_data"):
        """
        Inicializa el extractor de características.
        
        Args:
            output_dir (str): Directorio donde guardar los datos extraídos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        
        
        self.pose_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        
        self.activity_mapping = {
            "caminar_hacia": "caminarHacia",
            "caminar_regreso": "caminarRegreso", 
            "girar_90": "girar90",
            "girar_180": "girar180",
            "sentarse": "sentarse",
            "ponerse_de_pie": "ponerseDePie"
        }
        
        
        self.all_features = []
        self.extraction_stats = {
            "total_videos": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "activities_count": {},
            "average_frames_per_video": 0,
            "total_frames_processed": 0
        }

    def extract_from_video(self, video_path, activity_label, max_frames=None):
        """
        Extrae características de un video específico.
        
        Args:
            video_path (str): Ruta al archivo de video
            activity_label (str): Etiqueta de la actividad
            max_frames (int): Máximo número de frames a procesar (None para todos)
            
        Returns:
            list: Lista de diccionarios con características por frame
        """
        print(f"Procesando: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return []
        
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  - FPS: {fps:.2f}, Frames: {total_frames}, Duración: {duration:.2f}s")
        
        features_list = []
        frame_idx = 0
        successful_detections = 0
        
        
        landmark_history = []
        angle_history = []
        history_length = 5  
        
        pbar = tqdm(total=min(total_frames, max_frames or total_frames), 
                   desc=f"Procesando {Path(video_path).name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frame_idx >= max_frames:
                break
            
            
            results_dict, processed_frame = self.pose_detector.process_frame(frame, frame_idx)
            
            if results_dict and results_dict.get("landmarks"):
                
                landmarks = results_dict["landmarks"]
                angles = results_dict.get("angles", {})
                
                
                frame_features = {
                    
                    "video_path": str(video_path),
                    "activity": activity_label,
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else frame_idx,
                    
                    
                    **self._extract_landmark_features(landmarks),
                    
                    
                    **self._extract_angle_features(angles),
                    
                    
                    **self._extract_geometric_features(landmarks),
                }
                
                
                landmark_history.append(landmarks)
                angle_history.append(angles)
                
                
                if len(landmark_history) > history_length:
                    landmark_history.pop(0)
                    angle_history.pop(0)
                
                
                if len(landmark_history) >= 2:
                    temporal_features = self._extract_temporal_features(
                        landmark_history, angle_history, fps
                    )
                    frame_features.update(temporal_features)
                
                features_list.append(frame_features)
                successful_detections += 1
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"  - Detecciones exitosas: {successful_detections}/{frame_idx}")
        print(f"  - Tasa de éxito: {successful_detections/frame_idx*100:.1f}%")
        
        return features_list

    def _extract_landmark_features(self, landmarks):
        """
        Extrae características básicas de los landmarks.
        
        Args:
            landmarks (dict): Diccionario con landmarks detectados
            
        Returns:
            dict: Características extraídas de landmarks
        """
        features = {}
        
        
        key_points = [
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_wrist", "right_wrist"
        ]
        
        for point in key_points:
            if point in landmarks:
                features[f"{point}_x"] = landmarks[point]["x"]
                features[f"{point}_y"] = landmarks[point]["y"]
                features[f"{point}_z"] = landmarks[point]["z"]
                features[f"{point}_visibility"] = landmarks[point]["visibility"]
            else:
                
                features[f"{point}_x"] = 0.0
                features[f"{point}_y"] = 0.0
                features[f"{point}_z"] = 0.0
                features[f"{point}_visibility"] = 0.0
        
        return features

    def _extract_angle_features(self, angles):
        """
        Extrae características de ángulos articulares.
        
        Args:
            angles (dict): Diccionario con ángulos calculados
            
        Returns:
            dict: Características de ángulos
        """
        features = {}
        
        
        angle_keys = [
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle", 
            "left_elbow_angle", "right_elbow_angle",
            "torso_lean_angle", "shoulder_angle"
        ]
        
        for angle_key in angle_keys:
            if angle_key in angles:
                features[angle_key] = angles[angle_key]
            else:
                features[angle_key] = 0.0  
        
        
        if "left_knee_angle" in angles and "right_knee_angle" in angles:
            features["avg_knee_angle"] = (angles["left_knee_angle"] + angles["right_knee_angle"]) / 2
            features["knee_angle_diff"] = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
        
        return features

    def _extract_geometric_features(self, landmarks):
        """
        Extrae características geométricas derivadas.
        
        Args:
            landmarks (dict): Diccionario con landmarks
            
        Returns:
            dict: Características geométricas
        """
        features = {}
        
        
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            shoulder_dist = np.sqrt(
                (landmarks["left_shoulder"]["x"] - landmarks["right_shoulder"]["x"])**2 +
                (landmarks["left_shoulder"]["y"] - landmarks["right_shoulder"]["y"])**2 +
                (landmarks["left_shoulder"]["z"] - landmarks["right_shoulder"]["z"])**2
            )
            features["shoulder_width"] = shoulder_dist
        
        if "left_hip" in landmarks and "right_hip" in landmarks:
            hip_dist = np.sqrt(
                (landmarks["left_hip"]["x"] - landmarks["right_hip"]["x"])**2 +
                (landmarks["left_hip"]["y"] - landmarks["right_hip"]["y"])**2 +
                (landmarks["left_hip"]["z"] - landmarks["right_hip"]["z"])**2
            )
            features["hip_width"] = hip_dist
        
        
        if all(k in landmarks for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            center_x = (landmarks["left_shoulder"]["x"] + landmarks["right_shoulder"]["x"] + 
                       landmarks["left_hip"]["x"] + landmarks["right_hip"]["x"]) / 4
            center_y = (landmarks["left_shoulder"]["y"] + landmarks["right_shoulder"]["y"] + 
                       landmarks["left_hip"]["y"] + landmarks["right_hip"]["y"]) / 4
            center_z = (landmarks["left_shoulder"]["z"] + landmarks["right_shoulder"]["z"] + 
                       landmarks["left_hip"]["z"] + landmarks["right_hip"]["z"]) / 4
            
            features["body_center_x"] = center_x
            features["body_center_y"] = center_y
            features["body_center_z"] = center_z
        
        return features

    def _extract_temporal_features(self, landmark_history, angle_history, fps):
        """
        Extrae características temporales basadas en el historial de frames.
        
        Args:
            landmark_history (list): Historial de landmarks
            angle_history (list): Historial de ángulos
            fps (float): Frames por segundo del video
            
        Returns:
            dict: Características temporales
        """
        features = {}
        
        if len(landmark_history) < 2:
            return features
        
        
        current_landmarks = landmark_history[-1]
        previous_landmarks = landmark_history[-2]
        
        dt = 1.0 / fps if fps > 0 else 1.0  
        
        
        key_points = ["nose", "left_hip", "right_hip", "left_knee", "right_knee"]
        
        for point in key_points:
            if point in current_landmarks and point in previous_landmarks:
                
                vx = (current_landmarks[point]["x"] - previous_landmarks[point]["x"]) / dt
                vy = (current_landmarks[point]["y"] - previous_landmarks[point]["y"]) / dt
                vz = (current_landmarks[point]["z"] - previous_landmarks[point]["z"]) / dt
                
                
                v_total = np.sqrt(vx**2 + vy**2 + vz**2)
                
                features[f"{point}_velocity_x"] = vx
                features[f"{point}_velocity_y"] = vy
                features[f"{point}_velocity_z"] = vz
                features[f"{point}_velocity_total"] = v_total
        
        
        if len(angle_history) >= 2:
            current_angles = angle_history[-1]
            previous_angles = angle_history[-2]
            
            for angle_key in ["left_knee_angle", "right_knee_angle"]:
                if angle_key in current_angles and angle_key in previous_angles:
                    angular_velocity = (current_angles[angle_key] - previous_angles[angle_key]) / dt
                    features[f"{angle_key}_velocity"] = angular_velocity
        
        
        if len(landmark_history) >= 3:
            
            older_landmarks = landmark_history[-3]
            
            for point in key_points:
                if (point in current_landmarks and 
                    point in previous_landmarks and 
                    point in older_landmarks):
                    
                    
                    vx_current = (current_landmarks[point]["x"] - previous_landmarks[point]["x"]) / dt
                    vy_current = (current_landmarks[point]["y"] - previous_landmarks[point]["y"]) / dt
                    vz_current = (current_landmarks[point]["z"] - previous_landmarks[point]["z"]) / dt
                    
                    
                    vx_previous = (previous_landmarks[point]["x"] - older_landmarks[point]["x"]) / dt
                    vy_previous = (previous_landmarks[point]["y"] - older_landmarks[point]["y"]) / dt
                    vz_previous = (previous_landmarks[point]["z"] - older_landmarks[point]["z"]) / dt
                    
                    
                    ax = (vx_current - vx_previous) / dt
                    ay = (vy_current - vy_previous) / dt
                    az = (vz_current - vz_previous) / dt
                    
                    features[f"{point}_acceleration_x"] = ax
                    features[f"{point}_acceleration_y"] = ay
                    features[f"{point}_acceleration_z"] = az
                    features[f"{point}_acceleration_total"] = np.sqrt(ax**2 + ay**2 + az**2)
        
        return features

    def process_activity_folder(self, video_dir, activity_folder, max_videos=None, max_frames_per_video=None):
        """
        Procesa todos los videos en una carpeta de actividad específica.
        
        Args:
            video_dir (str): Directorio base de videos
            activity_folder (str): Nombre de la carpeta de actividad
            max_videos (int): Máximo número de videos a procesar por actividad
            max_frames_per_video (int): Máximo número de frames por video
            
        Returns:
            list: Lista con todas las características extraídas
        """
        activity_path = Path(video_dir) / activity_folder
        if not activity_path.exists():
            print(f"Advertencia: La carpeta {activity_path} no existe")
            return []
        
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(activity_path.glob(f"*{ext}"))
        
        if not video_files:
            print(f"No se encontraron videos en {activity_path}")
            return []
        
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        print(f"\nProcesando actividad: {activity_folder}")
        print(f"Videos encontrados: {len(video_files)}")
        
        activity_label = self.activity_mapping.get(activity_folder, activity_folder)
        all_activity_features = []
        
        
        for video_file in video_files:
            self.extraction_stats["total_videos"] += 1
            
            try:
                video_features = self.extract_from_video(
                    video_file, 
                    activity_label,
                    max_frames_per_video
                )
                
                if video_features:
                    all_activity_features.extend(video_features)
                    self.extraction_stats["successful_extractions"] += 1
                    self.extraction_stats["total_frames_processed"] += len(video_features)
                    
                    
                    video_name = video_file.stem
                    output_file = self.output_dir / "raw" / f"{activity_folder}_{video_name}_features.csv"
                    
                    df = pd.DataFrame(video_features)
                    df.to_csv(output_file, index=False)
                    print(f"  - Guardado: {output_file}")
                else:
                    self.extraction_stats["failed_extractions"] += 1
                    print(f"  - No se pudieron extraer características de {video_file}")
                    
            except Exception as e:
                self.extraction_stats["failed_extractions"] += 1
                print(f"  - Error procesando {video_file}: {str(e)}")
        
        
        if activity_label not in self.extraction_stats["activities_count"]:
            self.extraction_stats["activities_count"][activity_label] = 0
        self.extraction_stats["activities_count"][activity_label] += len(all_activity_features)
        
        print(f"Características extraídas para {activity_folder}: {len(all_activity_features)}")
        
        return all_activity_features

    def process_all_videos(self, video_dir="videos", max_videos_per_activity=None, max_frames_per_video=None):
        """
        Procesa todos los videos en todas las carpetas de actividades.
        
        Args:
            video_dir (str): Directorio que contiene las carpetas de actividades
            max_videos_per_activity (int): Máximo videos por actividad
            max_frames_per_video (int): Máximo frames por video
        """
        print("=== Iniciando extracción de características ===")
        print(f"Directorio de videos: {video_dir}")
        print(f"Directorio de salida: {self.output_dir}")
        
        video_dir_path = Path(video_dir)
        if not video_dir_path.exists():
            raise FileNotFoundError(f"El directorio {video_dir} no existe")
        
        
        activity_folders = [d.name for d in video_dir_path.iterdir() 
                           if d.is_dir() and d.name in self.activity_mapping]
        
        if not activity_folders:
            raise ValueError(f"No se encontraron carpetas de actividades válidas en {video_dir}")
        
        print(f"Actividades encontradas: {activity_folders}")
        
        
        for activity_folder in activity_folders:
            activity_features = self.process_activity_folder(
                video_dir, 
                activity_folder,
                max_videos_per_activity,
                max_frames_per_video
            )
            self.all_features.extend(activity_features)
        
        
        if self.all_features:
            self._save_complete_dataset()
            self._generate_statistics()
            print("\n=== Extracción completada exitosamente ===")
        else:
            print("\n=== No se extrajeron características ===")

    def _save_complete_dataset(self):
        """Guarda el dataset completo con todas las características."""
        print("\nGuardando dataset completo...")
        
        
        df = pd.DataFrame(self.all_features)
        
        
        main_output = self.output_dir / "complete_dataset.csv"
        df.to_csv(main_output, index=False)
        print(f"Dataset completo guardado: {main_output}")
        
        
        feature_columns = [col for col in df.columns 
                          if col not in ["video_path", "frame_idx", "timestamp"]]
        
        processed_df = df[feature_columns]
        processed_output = self.output_dir / "processed" / "training_features.csv"
        processed_df.to_csv(processed_output, index=False)
        print(f"Características de entrenamiento guardadas: {processed_output}")
        
        
        for activity in df["activity"].unique():
            activity_df = df[df["activity"] == activity]
            activity_output = self.output_dir / "processed" / f"{activity}_features.csv"
            activity_df.to_csv(activity_output, index=False)
            print(f"Características de {activity}: {activity_output}")

    def _generate_statistics(self):
        """Genera estadísticas del dataset extraído."""
        print("\nGenerando estadísticas...")
        
        if not self.all_features:
            return
        
        df = pd.DataFrame(self.all_features)
        
        
        stats = {
            "extraction_info": self.extraction_stats,
            "dataset_info": {
                "total_samples": len(df),
                "total_features": len(df.columns),
                "activities": list(df["activity"].unique()),
                "samples_per_activity": df["activity"].value_counts().to_dict(),
            },
            "feature_statistics": {}
        }
        
        
        if self.extraction_stats["successful_extractions"] > 0:
            stats["extraction_info"]["average_frames_per_video"] = (
                self.extraction_stats["total_frames_processed"] / 
                self.extraction_stats["successful_extractions"]
            )
        
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ["frame_idx", "timestamp"]:
                stats["feature_statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "missing_values": int(df[col].isna().sum())
                }
        
        
        stats_output = self.output_dir / "statistics" / "extraction_stats.json"
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Estadísticas guardadas: {stats_output}")
        
        
        print("\n=== RESUMEN DE EXTRACCIÓN ===")
        print(f"Videos procesados: {self.extraction_stats['successful_extractions']}/{self.extraction_stats['total_videos']}")
        print(f"Total de muestras extraídas: {len(df)}")
        print(f"Promedio de frames por video: {stats['extraction_info']['average_frames_per_video']:.1f}")
        print("Muestras por actividad:")
        for activity, count in stats["dataset_info"]["samples_per_activity"].items():
            print(f"  - {activity}: {count}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Extractor de características de videos para sistema de anotación")
    parser.add_argument("--video_dir", default="videos", help="Directorio que contiene los videos por actividad")
    parser.add_argument("--output_dir", default="data/training_data", help="Directorio de salida para los datos")
    parser.add_argument("--max_videos", type=int, help="Máximo número de videos por actividad")
    parser.add_argument("--max_frames", type=int, help="Máximo número de frames por video")
    
    args = parser.parse_args()
    
    
    extractor = VideoFeatureExtractor(args.output_dir)
    
    try:
        
        extractor.process_all_videos(
            video_dir=args.video_dir,
            max_videos_per_activity=args.max_videos,
            max_frames_per_video=args.max_frames
        )
        
        print("\n¡Extracción de características completada exitosamente!")
        print(f"Los datos están disponibles en: {args.output_dir}")
        print("\nPróximos pasos:")
        print("1. Revisar las estadísticas en: data/training_data/statistics/")
        print("2. Entrenar modelos con: python train_classifier.py")
        
    except Exception as e:
        print(f"Error durante la extracción: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 