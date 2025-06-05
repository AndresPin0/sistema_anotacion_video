import numpy as np
import time

class SimpleActivityClassifier:
    """
    Clasificador simple de actividades basado en reglas.
    """
    
    def __init__(self):
        """Inicializa el clasificador"""
        self.activities = [
            "caminarHacia", 
            "caminarRegreso", 
            "girar90", 
            "girar180", 
            "sentarse", 
            "ponerseDePie"
        ]
        
        self.prediction_history = []
        self.history_length = 2  
        
        self.last_prediction = "desconocida"
        self.last_landmarks = None
        self.frame_counter = 0
        self.last_prediction_time = time.time()
        
        self.position_history = []
        self.position_history_length = 3


        self.thresholds = {
            "z_threshold": 1,
            "knee_angle_threshold": 175,
            "hip_movement_threshold": 0.0005,
            "orientation_threshold": 5,
            "orientation_threshold_large": 30 
        }
        
        self.debug_mode = True
        self.debug_all_frames = True
        self.last_debug_frame = 0
        self.debug_frequency = 1
    
    def predict(self, landmarks_dict, angles_dict):
        """
        Predice la actividad basándose en los landmarks y ángulos.
        
        Args:
            landmarks_dict: Diccionario con los landmarks detectados
            angles_dict: Diccionario con los ángulos de las articulaciones
            
        Returns:
            str: Actividad predicha
        """
        
        if not landmarks_dict or not angles_dict:
            print("ADVERTENCIA: Sin landmarks o ángulos para procesar.")
            return "desconocida"
        
        current_time = time.time()
        self.frame_counter += 1
        
        if "nose" in landmarks_dict and "left_hip" in landmarks_dict and "right_hip" in landmarks_dict:
            self.position_history.append({
                "nose_z": landmarks_dict["nose"]["z"],
                "hip_y": (landmarks_dict['left_hip']['y'] + landmarks_dict['right_hip']['y']) / 2,
                "timestamp": current_time
            })
            if len(self.position_history) > self.position_history_length:
                self.position_history.pop(0)
        
        prediction = self._apply_rules(landmarks_dict, angles_dict)
        
        if self.debug_mode and self.debug_all_frames:
            self._print_debug_info(landmarks_dict, angles_dict, prediction)
        
        elif self.debug_mode and (self.frame_counter - self.last_debug_frame) >= self.debug_frequency:
            self._print_debug_info(landmarks_dict, angles_dict, prediction)
            self.last_debug_frame = self.frame_counter
        
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        if self.prediction_history:
            active_predictions = [p for p in self.prediction_history if p != "desconocida"]
            if active_predictions:
                unique_predictions, counts = np.unique(active_predictions, return_counts=True)
                smoothed_prediction = unique_predictions[np.argmax(counts)]
            else:
                unique_predictions, counts = np.unique(self.prediction_history, return_counts=True)
                smoothed_prediction = unique_predictions[np.argmax(counts)]
        else:
            smoothed_prediction = prediction
        

        if smoothed_prediction != self.last_prediction:
            time_since_last = current_time - self.last_prediction_time
            print(f"CAMBIO DE ACTIVIDAD: {self.last_prediction} -> {smoothed_prediction} (después de {time_since_last:.2f}s)")
            self.last_prediction_time = current_time
            
            if self.frame_counter > 30 and smoothed_prediction == "desconocida" and self.last_prediction == "desconocida":
                test_activities = ["caminarHacia", "caminarRegreso", "girar90", "girar180", "sentarse", "ponerseDePie"]
                test_idx = (self.frame_counter // 30) % len(test_activities)
                print(f"FORZANDO ACTIVIDAD DE PRUEBA: {test_activities[test_idx]}")
        
        self.last_prediction = smoothed_prediction
        self.last_landmarks = landmarks_dict
        
        return smoothed_prediction
    
    def _apply_rules(self, landmarks, angles):
        """
        Aplica reglas simples para predecir la actividad.
        
        Args:
            landmarks: Diccionario con landmarks detectados
            angles: Diccionario con ángulos de articulaciones
            
        Returns:
            str: Actividad predicha según las reglas
        """
        nose = landmarks.get('nose', {})
        nose_z = nose.get('z', 0)
        

        z_velocity = 0
        hip_y_velocity = 0
        time_diff = 1
        
        if len(self.position_history) >= 2:
            current = self.position_history[-1]
            previous = self.position_history[0]
            time_diff = current["timestamp"] - previous["timestamp"]
            if time_diff > 0:  
                z_velocity = (current["nose_z"] - previous["nose_z"]) / time_diff
                hip_y_velocity = (current["hip_y"] - previous["hip_y"]) / time_diff
        
        left_knee_angle = angles.get('left_knee_angle', 180)
        right_knee_angle = angles.get('right_knee_angle', 180)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        activities_detected = []
        
        
        
        if 'nose' in landmarks and 'z' in landmarks['nose']:
            
            if nose_z < -self.thresholds["z_threshold"] or z_velocity < -self.thresholds["z_threshold"]:
                activities_detected.append(("caminarHacia", 5, abs(nose_z) + abs(z_velocity)))  
            elif nose_z > self.thresholds["z_threshold"] or z_velocity > self.thresholds["z_threshold"]:
                activities_detected.append(("caminarRegreso", 5, abs(nose_z) + abs(z_velocity)))
        
        
        
        if self.last_landmarks and 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            
            shoulder_vector_current = {
                'x': landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'],
                'y': landmarks['left_shoulder']['y'] - landmarks['right_shoulder']['y']
            }
            
            shoulder_vector_prev = {
                'x': self.last_landmarks['left_shoulder']['x'] - self.last_landmarks['right_shoulder']['x'],
                'y': self.last_landmarks['left_shoulder']['y'] - self.last_landmarks['right_shoulder']['y']
            }
            
            
            orientation_change = self._calculate_angle_between_vectors(
                shoulder_vector_current,
                shoulder_vector_prev
            )
            
            if orientation_change > self.thresholds["orientation_threshold"]:
                
                if orientation_change > self.thresholds["orientation_threshold_large"]:
                    activities_detected.append(("girar180", 8, orientation_change))  
                else:
                    activities_detected.append(("girar90", 6, orientation_change))
        
        
        
        if avg_knee_angle < self.thresholds["knee_angle_threshold"]:
            
            if hip_y_velocity > self.thresholds["hip_movement_threshold"]:
                
                activities_detected.append(("sentarse", 7, hip_y_velocity * 100))
            elif hip_y_velocity < -self.thresholds["hip_movement_threshold"]:
                
                activities_detected.append(("ponerseDePie", 7, abs(hip_y_velocity) * 100))
        
        
        if activities_detected:
            
            activities_detected.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return activities_detected[0][0]
        
        
        return "desconocida"
    
    def _calculate_angle_between_vectors(self, vector1, vector2):
        """
        Calcula el ángulo entre dos vectores 2D.
        
        Args:
            vector1, vector2: Diccionarios con componentes x, y
            
        Returns:
            float: Ángulo en grados
        """
        mag1 = (vector1['x']**2 + vector1['y']**2)**0.5
        mag2 = (vector2['x']**2 + vector2['y']**2)**0.5
        
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0
        
        dot_product = vector1['x'] * vector2['x'] + vector1['y'] * vector2['y']
        
        cos_angle = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _print_debug_info(self, landmarks, angles, prediction):
        """Imprime información de depuración sobre el estado actual"""
        print("\n=== INFORMACIÓN DE DEPURACIÓN Frame", self.frame_counter, "===")
        print(f"Predicción actual RAW: {prediction}")
        print(f"Historial de predicciones: {self.prediction_history}")
        
        
        if 'nose' in landmarks and 'z' in landmarks['nose']:
            nose_z = landmarks['nose']['z']
            
            
            z_velocity = 0
            if len(self.position_history) >= 2:
                current = self.position_history[-1]
                previous = self.position_history[0]
                time_diff = current["timestamp"] - previous["timestamp"]
                if time_diff > 0:
                    z_velocity = (current["nose_z"] - previous["nose_z"]) / time_diff
            
            print(f"CAMINAR - Nariz Z: {nose_z:.4f}, Velocidad Z: {z_velocity:.4f} (Umbral: ±{self.thresholds['z_threshold']:.4f})")
            print(f"  • caminarHacia: Z < -{self.thresholds['z_threshold']} o velocidad Z < -{self.thresholds['z_threshold']}")
            print(f"  • caminarRegreso: Z > {self.thresholds['z_threshold']} o velocidad Z > {self.thresholds['z_threshold']}")
            
            
            if abs(nose_z) > self.thresholds['z_threshold'] or abs(z_velocity) > self.thresholds['z_threshold']:
                color = '\033[92m'  
                direction = "HACIA CÁMARA" if (nose_z < -self.thresholds['z_threshold'] or z_velocity < -self.thresholds['z_threshold']) else "ALEJARSE"
                print(f"{color}¡DEBERÍA DETECTAR CAMINAR {direction}! Z={nose_z:.4f}, Velocidad={z_velocity:.4f}\033[0m")
        
        
        if "left_knee_angle" in angles and "right_knee_angle" in angles:
            left_knee = angles["left_knee_angle"]
            right_knee = angles["right_knee_angle"]
            avg_knee = (left_knee + right_knee) / 2
            
            
            hip_y_velocity = 0
            if len(self.position_history) >= 2:
                current = self.position_history[-1]
                previous = self.position_history[0]
                time_diff = current["timestamp"] - previous["timestamp"]
                if time_diff > 0:
                    hip_y_velocity = (current["hip_y"] - previous["hip_y"]) / time_diff
            
            print(f"SENTARSE/LEVANTARSE - Ángulo promedio rodillas: {avg_knee:.2f}° (Umbral: < {self.thresholds['knee_angle_threshold']}°)")
            print(f"  • Movimiento vertical cadera: {hip_y_velocity:.6f} (Umbral: ±{self.thresholds['hip_movement_threshold']:.6f})")
            
            if avg_knee < self.thresholds['knee_angle_threshold']:
                color = '\033[92m'  
                print(f"{color}¡RODILLAS FLEXIONADAS! Ángulo={avg_knee:.2f}°\033[0m")
                
                if abs(hip_y_velocity) > self.thresholds['hip_movement_threshold']:
                    direction = "ABAJO (SENTARSE)" if hip_y_velocity > 0 else "ARRIBA (LEVANTARSE)"
                    print(f"{color}¡MOVIMIENTO CADERA {direction}! Velocidad={hip_y_velocity:.6f}\033[0m")
        
        
        if self.last_landmarks and 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            shoulder_vector_current = {
                'x': landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'],
                'y': landmarks['left_shoulder']['y'] - landmarks['right_shoulder']['y']
            }
            shoulder_vector_prev = {
                'x': self.last_landmarks['left_shoulder']['x'] - self.last_landmarks['right_shoulder']['x'],
                'y': self.last_landmarks['left_shoulder']['y'] - self.last_landmarks['right_shoulder']['y']
            }
            orientation_change = self._calculate_angle_between_vectors(
                shoulder_vector_current, shoulder_vector_prev
            )
            print(f"GIRAR - Cambio de orientación: {orientation_change:.2f}° (Umbral 90°: > {self.thresholds['orientation_threshold']}°, Umbral 180°: > {self.thresholds['orientation_threshold_large']}°)")
            
            if orientation_change > self.thresholds['orientation_threshold']:
                color = '\033[92m'  
                giro_tipo = "GRANDE (180°)" if orientation_change > self.thresholds['orientation_threshold_large'] else "PEQUEÑO (90°)"
                print(f"{color}¡GIRO {giro_tipo} DETECTADO! Ángulo={orientation_change:.2f}°\033[0m")
        
        print("===============================")


if __name__ == "__main__":
    classifier = SimpleActivityClassifier()
    
    test_landmarks = {
        'nose': {'x': 0.1, 'y': -0.2, 'z': -0.1, 'visibility': 0.98},
        'left_shoulder': {'x': -0.2, 'y': 0.0, 'z': -0.05, 'visibility': 0.98},
        'right_shoulder': {'x': 0.2, 'y': 0.0, 'z': -0.05, 'visibility': 0.98},
        'left_hip': {'x': -0.1, 'y': 0.3, 'z': -0.05, 'visibility': 0.9},
        'right_hip': {'x': 0.1, 'y': 0.3, 'z': -0.05, 'visibility': 0.9},
        'left_knee': {'x': -0.15, 'y': 0.6, 'z': -0.05, 'visibility': 0.9},
        'right_knee': {'x': 0.15, 'y': 0.6, 'z': -0.05, 'visibility': 0.9},
        'left_ankle': {'x': -0.15, 'y': 0.9, 'z': -0.05, 'visibility': 0.8},
        'right_ankle': {'x': 0.15, 'y': 0.9, 'z': -0.05, 'visibility': 0.8},
        'left_wrist': {'x': -0.4, 'y': 0.3, 'z': -0.05, 'visibility': 0.95},
        'right_wrist': {'x': 0.4, 'y': 0.3, 'z': -0.05, 'visibility': 0.95},
        'left_elbow': {'x': -0.3, 'y': 0.15, 'z': -0.05, 'visibility': 0.95},
        'right_elbow': {'x': 0.3, 'y': 0.15, 'z': -0.05, 'visibility': 0.95}
    }
    
    test_angles = {
        'left_knee_angle': 175.0,
        'right_knee_angle': 175.0,
        'left_hip_angle': 170.0,
        'right_hip_angle': 170.0,
        'left_elbow_angle': 160.0,
        'right_elbow_angle': 160.0,
        'trunk_lateral_inclination': 5.0
    }
    
    prediction = classifier.predict(test_landmarks, test_angles)
    print(f"Actividad predicha: {prediction}") 