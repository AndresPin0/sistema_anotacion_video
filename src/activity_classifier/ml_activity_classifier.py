import os
import sys
import numpy as np
import joblib
from pathlib import Path
import time

# Agregar el directorio raíz del proyecto al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class MLActivityClassifier:
    """
    Clasificador de actividades usando modelos de machine learning entrenados.
    """
    
    def __init__(self, models_dir="models"):
        """
        Inicializa el clasificador ML.
        
        Args:
            models_dir (str): Directorio con los modelos entrenados
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Historial para características temporales
        self.landmark_history = []
        self.angle_history = []
        self.history_length = 5
        
        # Configuración de predicción
        self.prediction_history = []
        self.prediction_history_length = 3
        
        # Cargar modelo y preprocesadores
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Carga el mejor modelo entrenado y sus preprocesadores."""
        try:
            # Buscar información del mejor modelo
            best_model_file = self.models_dir / "evaluation" / "best_model_info.json"
            if best_model_file.exists():
                import json
                with open(best_model_file, 'r') as f:
                    best_model_info = json.load(f)
                model_name = best_model_info['best_model']
                print(f"Cargando mejor modelo: {model_name}")
            else:
                # Si no hay info del mejor modelo, usar Random Forest por defecto
                model_name = "RandomForest"
                print("Usando modelo por defecto: RandomForest")
            
            # Cargar modelo
            model_path = self.models_dir / "trained_models" / f"{model_name}.joblib"
            if model_path.exists():
                self.model = joblib.load(model_path)
                print(f"✓ Modelo cargado: {model_path}")
            else:
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
            # Cargar scaler
            scaler_path = self.models_dir / "trained_models" / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler cargado: {scaler_path}")
            
            # Cargar label encoder
            encoder_path = self.models_dir / "trained_models" / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print(f"✓ Label encoder cargado: {encoder_path}")
            
            # Cargar nombres de características
            features_path = self.models_dir / "feature_analysis" / "selected_features.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    feature_info = json.load(f)
                self.feature_names = feature_info['selected_features']
                print(f"✓ Características cargadas: {len(self.feature_names)} features")
            
            print("🎉 Clasificador ML inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            print("Fallback: Usando clasificador simple...")
            self.model = None
    
    def predict(self, landmarks_dict, angles_dict):
        """
        Predice la actividad usando el modelo entrenado.
        
        Args:
            landmarks_dict: Diccionario con landmarks detectados
            angles_dict: Diccionario con ángulos articulares
            
        Returns:
            str: Actividad predicha
        """
        if self.model is None:
            # Fallback al clasificador simple si no hay modelo
            from .activity_classifier import SimpleActivityClassifier
            if not hasattr(self, '_simple_classifier'):
                self._simple_classifier = SimpleActivityClassifier()
            return self._simple_classifier.predict(landmarks_dict, angles_dict)
        
        try:
            # Extraer características igual que en el entrenamiento
            features = self._extract_features(landmarks_dict, angles_dict)
            
            if features is None:
                return "desconocida"
            
            # Normalizar características
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Predecir
            prediction_encoded = self.model.predict(features_scaled)[0]
            
            # Decodificar etiqueta
            if self.label_encoder:
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            else:
                prediction = str(prediction_encoded)
            
            # Suavizar predicción con historial
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > self.prediction_history_length:
                self.prediction_history.pop(0)
            
            # Usar la predicción más común en el historial
            if len(self.prediction_history) > 1:
                unique_preds, counts = np.unique(self.prediction_history, return_counts=True)
                smoothed_prediction = unique_preds[np.argmax(counts)]
            else:
                smoothed_prediction = prediction
            
            return smoothed_prediction
            
        except Exception as e:
            print(f"Error en predicción ML: {str(e)}")
            return "desconocida"
    
    def _extract_features(self, landmarks, angles):
        """
        Extrae las mismas características que se usaron en el entrenamiento.
        
        Args:
            landmarks: Diccionario con landmarks
            angles: Diccionario con ángulos
            
        Returns:
            list: Vector de características
        """
        if not landmarks or not angles:
            return None
        
        # Agregar al historial
        self.landmark_history.append(landmarks)
        self.angle_history.append(angles)
        
        # Mantener solo los últimos N frames
        if len(self.landmark_history) > self.history_length:
            self.landmark_history.pop(0)
            self.angle_history.pop(0)
        
        features = {}
        
        # 1. Características de landmarks (coordenadas normalizadas)
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
        
        # 2. Características de ángulos
        angle_keys = [
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle", 
            "left_elbow_angle", "right_elbow_angle"
        ]
        
        for angle_key in angle_keys:
            if angle_key in angles:
                features[angle_key] = angles[angle_key]
            else:
                features[angle_key] = 0.0
        
        # Características derivadas
        if "left_knee_angle" in angles and "right_knee_angle" in angles:
            features["avg_knee_angle"] = (angles["left_knee_angle"] + angles["right_knee_angle"]) / 2
            features["knee_angle_diff"] = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
        
        # 3. Características geométricas
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
        
        # 4. Características temporales (si hay suficiente historial)
        if len(self.landmark_history) >= 2:
            current_landmarks = self.landmark_history[-1]
            previous_landmarks = self.landmark_history[-2]
            
            # Velocidades de puntos clave
            key_points_temporal = ["nose", "left_hip", "right_hip", "left_knee", "right_knee"]
            
            for point in key_points_temporal:
                if point in current_landmarks and point in previous_landmarks:
                    vx = current_landmarks[point]["x"] - previous_landmarks[point]["x"]
                    vy = current_landmarks[point]["y"] - previous_landmarks[point]["y"]
                    vz = current_landmarks[point]["z"] - previous_landmarks[point]["z"]
                    v_total = np.sqrt(vx**2 + vy**2 + vz**2)
                    
                    features[f"{point}_velocity_x"] = vx
                    features[f"{point}_velocity_y"] = vy
                    features[f"{point}_velocity_z"] = vz
                    features[f"{point}_velocity_total"] = v_total
        
        # 5. Aceleraciones (si hay suficiente historial)
        if len(self.landmark_history) >= 3:
            current = self.landmark_history[-1]
            previous = self.landmark_history[-2]
            older = self.landmark_history[-3]
            
            for point in ["nose", "left_hip", "right_hip"]:
                if all(point in lm for lm in [current, previous, older]):
                    # Velocidad actual
                    vx_curr = current[point]["x"] - previous[point]["x"]
                    vy_curr = current[point]["y"] - previous[point]["y"]
                    vz_curr = current[point]["z"] - previous[point]["z"]
                    
                    # Velocidad anterior
                    vx_prev = previous[point]["x"] - older[point]["x"]
                    vy_prev = previous[point]["y"] - older[point]["y"]
                    vz_prev = previous[point]["z"] - older[point]["z"]
                    
                    # Aceleración
                    ax = vx_curr - vx_prev
                    ay = vy_curr - vy_prev
                    az = vz_curr - vz_prev
                    
                    features[f"{point}_acceleration_x"] = ax
                    features[f"{point}_acceleration_y"] = ay
                    features[f"{point}_acceleration_z"] = az
                    features[f"{point}_acceleration_total"] = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Filtrar solo las características que se usaron en el entrenamiento
        if self.feature_names:
            filtered_features = []
            for feature_name in self.feature_names:
                filtered_features.append(features.get(feature_name, 0.0))
            return filtered_features
        else:
            # Si no tenemos los nombres, usar todas las características
            return list(features.values())
    
    def get_prediction_confidence(self):
        """Retorna la confianza de la predicción basada en el historial."""
        if not self.prediction_history:
            return 0.0
        
        # Calcular estabilidad de las predicciones
        unique_preds, counts = np.unique(self.prediction_history, return_counts=True)
        max_count = np.max(counts)
        confidence = max_count / len(self.prediction_history) * 100
        
        return confidence 