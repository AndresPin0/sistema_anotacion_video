
"""
Clasificador de Actividades Mejorado
==================================

Este módulo implementa un clasificador de actividades mejorado usando
XGBoost y técnicas de ensemble para mayor precisión.
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

class ActivityClassifier:
    """Clasificador de actividades mejorado con XGBoost y técnicas de ensemble."""
    
    ACTIVITIES = [
        'caminar_hacia',
        'caminar_regreso',
        'girar_90',
        'girar_180',
        'sentarse',
        'ponerse_pie'
    ]
    
    def __init__(self):
        """Inicializa el clasificador de actividades."""
        self.model = None
        self.scaler = None
        self.detection_buffer = []  
        self.buffer_size = 5  
        self.confidence_threshold = 0.85  
        self.temporal_smoothing = True  
        self.load_models()
        
    def load_models(self, model_path: str = "models/trained_models") -> None:
        """
        Carga los modelos entrenados y el scaler.
        
        Args:
            model_path: Ruta al directorio con los modelos
        """
        model_dir = Path(model_path)
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(model_dir / "xgboost_model.json"))
            self.scaler = joblib.load(model_dir / "scaler.joblib")
            print("✓ Modelos cargados correctamente")
        except Exception as e:
            print(f"⚠️  Error cargando modelos: {e}")
            print("   Se usará clasificación basada en reglas como fallback")
    
    def preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Preprocesa las características para clasificación.
        
        Args:
            features: Diccionario de características extraídas
        Returns:
            np.ndarray: Características preprocesadas
        """
        
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
            
        return feature_vector
    
    def _rule_based_classification(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Clasificación basada en reglas como fallback.
        
        Args:
            features: Diccionario de características
        Returns:
            Tuple[str, float]: (actividad predicha, confianza)
        """
        
        
        
        if (features.get('vertical_movement', 0) < -0.15 and 
            features.get('trunk_forward_tilt', 90) > 100):
            return 'sentarse', 0.8
            
        
        if (features.get('vertical_movement', 0) > 0.15 and 
            features.get('trunk_forward_tilt', 90) < 80):
            return 'ponerse_pie', 0.8
            
        
        if abs(features.get('trunk_lateral_tilt', 0) - 90) > 45:
            if abs(features.get('trunk_lateral_tilt', 0) - 90) > 80:
                return 'girar_180', 0.75
            return 'girar_90', 0.75
            
        
        if abs(features.get('forward_movement', 0)) > 0.1:
            if features.get('forward_movement', 0) > 0:
                return 'caminar_hacia', 0.7
            return 'caminar_regreso', 0.7
            
        return 'desconocido', 0.0
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Clasifica la actividad basada en las características extraídas.
        
        Args:
            features: Diccionario de características extraídas
        Returns:
            Tuple[str, float]: (actividad predicha, confianza)
        """
        
        X = self.preprocess_features(features)
        
        try:
            if self.model is not None:
                
                dmatrix = xgb.DMatrix(X)
                probabilities = self.model.predict(dmatrix)
                
                
                pred_idx = np.argmax(probabilities)
                confidence = float(probabilities[pred_idx])
                
                if confidence >= self.confidence_threshold:
                    prediction = self.ACTIVITIES[pred_idx]
                else:
                    
                    prediction, confidence = self._rule_based_classification(features)
                    
            else:
                
                prediction, confidence = self._rule_based_classification(features)
                
        except Exception as e:
            print(f"⚠️  Error en clasificación: {e}")
            prediction, confidence = self._rule_based_classification(features)
        
        
        if self.temporal_smoothing:
            self.detection_buffer.append((prediction, confidence))
            if len(self.detection_buffer) > self.buffer_size:
                self.detection_buffer.pop(0)
            
            
            if len(self.detection_buffer) >= 3:
                predictions = [p for p, _ in self.detection_buffer]
                confidences = [c for _, c in self.detection_buffer]
                
                
                unique_preds, counts = np.unique(predictions, return_counts=True)
                max_count_idx = np.argmax(counts)
                
                if counts[max_count_idx] >= len(self.detection_buffer) * 0.6:
                    prediction = unique_preds[max_count_idx]
                    confidence = np.mean([c for p, c in self.detection_buffer if p == prediction])
        
        return prediction, confidence
    
    def set_parameters(self, 
                      confidence_threshold: Optional[float] = None,
                      buffer_size: Optional[int] = None,
                      temporal_smoothing: Optional[bool] = None) -> None:
        """
        Actualiza los parámetros del clasificador.
        
        Args:
            confidence_threshold: Nuevo umbral de confianza
            buffer_size: Nuevo tamaño del buffer temporal
            temporal_smoothing: Habilitar/deshabilitar suavizado temporal
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if buffer_size is not None:
            self.buffer_size = buffer_size
            self.detection_buffer = self.detection_buffer[-buffer_size:]
        if temporal_smoothing is not None:
            self.temporal_smoothing = temporal_smoothing 