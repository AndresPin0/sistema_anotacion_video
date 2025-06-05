import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

class PoseDetector:
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Inicializa el detector de poses de MediaPipe.
        
        Args:
            static_image_mode: Si es True, trata cada imagen de forma independiente
            model_complexity: Complejidad del modelo (0, 1 o 2)
            smooth_landmarks: Si es True, filtra los landmarks para reducir el ruido
            min_detection_confidence: Confianza mínima para detección inicial
            min_tracking_confidence: Confianza mínima para seguimiento
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        
        self.landmarks_of_interest = [
            
            self.mp_pose.PoseLandmark.NOSE,
            
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW
        ]
        
        
        self.landmark_names = {
            0: "nose",
            11: "left_shoulder", 12: "right_shoulder",
            23: "left_hip", 24: "right_hip",
            25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle",
            15: "left_wrist", 16: "right_wrist",
            13: "left_elbow", 14: "right_elbow"
        }
        
        
        self.custom_connections = [
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
    
    def detect_pose(self, image):
        """
        Detecta la pose en una imagen.
        
        Args:
            image: Imagen en formato BGR (OpenCV)
            
        Returns:
            tuple: (results, processed_image) donde results son los resultados de MediaPipe
                  y processed_image es la imagen con los landmarks dibujados
        """
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        
        
        results = self.pose.process(image_rgb)
        
        
        annotated_image = image.copy()
        
        
        if results.pose_landmarks:
            
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return results, annotated_image
    
    def extract_landmarks(self, results, normalize=True):
        """
        Extrae las coordenadas de los landmarks de interés.
        
        Args:
            results: Resultados de la detección de pose de MediaPipe
            normalize: Si es True, normaliza las coordenadas respecto al centro de la cadera
            
        Returns:
            dict: Diccionario con las coordenadas (x, y, z, visibility) de cada landmark de interés
                 o None si no se detectó ninguna pose
        """
        if not results.pose_landmarks:
            return None
        
        landmarks_dict = {}
        
        
        if normalize:
            
            left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            hip_center_z = (left_hip.z + right_hip.z) / 2
            
            
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_distance = ((left_shoulder.x - right_shoulder.x)**2 + 
                                (left_shoulder.y - right_shoulder.y)**2 + 
                                (left_shoulder.z - right_shoulder.z)**2)**0.5
            scale_factor = 1.0 / max(shoulder_distance, 1e-6)  
        
        
        for landmark_idx in self.landmarks_of_interest:
            landmark = results.pose_landmarks.landmark[landmark_idx]
            
            if normalize:
                
                x_norm = (landmark.x - hip_center_x) * scale_factor
                y_norm = (landmark.y - hip_center_y) * scale_factor
                z_norm = (landmark.z - hip_center_z) * scale_factor
                
                landmarks_dict[self.landmark_names[landmark_idx]] = {
                    'x': x_norm,
                    'y': y_norm,
                    'z': z_norm,
                    'visibility': landmark.visibility
                }
            else:
                landmarks_dict[self.landmark_names[landmark_idx]] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return landmarks_dict
    
    def calculate_joint_angles(self, landmarks_dict):
        """
        Calcula los ángulos de las articulaciones a partir de los landmarks.
        
        Args:
            landmarks_dict: Diccionario con las coordenadas de los landmarks
            
        Returns:
            dict: Diccionario con los ángulos de las articulaciones de interés
        """
        if not landmarks_dict:
            return None
        
        angles = {}
        
        
        angles['left_knee_angle'] = self._calculate_angle(
            landmarks_dict['left_hip'],
            landmarks_dict['left_knee'],
            landmarks_dict['left_ankle']
        )
        
        
        angles['right_knee_angle'] = self._calculate_angle(
            landmarks_dict['right_hip'],
            landmarks_dict['right_knee'],
            landmarks_dict['right_ankle']
        )
        
        
        angles['left_hip_angle'] = self._calculate_angle(
            landmarks_dict['left_shoulder'],
            landmarks_dict['left_hip'],
            landmarks_dict['left_knee']
        )
        
        
        angles['right_hip_angle'] = self._calculate_angle(
            landmarks_dict['right_shoulder'],
            landmarks_dict['right_hip'],
            landmarks_dict['right_knee']
        )
        
        
        angles['left_elbow_angle'] = self._calculate_angle(
            landmarks_dict['left_shoulder'],
            landmarks_dict['left_elbow'],
            landmarks_dict['left_wrist']
        )
        
        
        angles['right_elbow_angle'] = self._calculate_angle(
            landmarks_dict['right_shoulder'],
            landmarks_dict['right_elbow'],
            landmarks_dict['right_wrist']
        )
        
        
        
        left_shoulder = landmarks_dict['left_shoulder']
        right_shoulder = landmarks_dict['right_shoulder']
        
        
        shoulder_vector = {
            'x': left_shoulder['x'] - right_shoulder['x'],
            'y': left_shoulder['y'] - right_shoulder['y'],
            'z': left_shoulder['z'] - right_shoulder['z']
        }
        
        
        vertical_vector = {'x': 0, 'y': 1, 'z': 0}
        
        
        shoulder_vector_2d = {'x': shoulder_vector['x'], 'y': shoulder_vector['y']}
        vertical_vector_2d = {'x': vertical_vector['x'], 'y': vertical_vector['y']}
        
        angles['trunk_lateral_inclination'] = self._calculate_angle_between_vectors(
            shoulder_vector_2d, vertical_vector_2d
        )
        
        return angles
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calcula el ángulo entre tres puntos 3D (p1-p2-p3).
        
        Args:
            point1, point2, point3: Diccionarios con las coordenadas x, y, z
            
        Returns:
            float: Ángulo en grados
        """
        
        vector1 = {
            'x': point1['x'] - point2['x'],
            'y': point1['y'] - point2['y'],
            'z': point1['z'] - point2['z']
        }
        
        
        vector2 = {
            'x': point3['x'] - point2['x'],
            'y': point3['y'] - point2['y'],
            'z': point3['z'] - point2['z']
        }
        
        return self._calculate_angle_between_vectors(vector1, vector2)
    
    def _calculate_angle_between_vectors(self, vector1, vector2):
        """
        Calcula el ángulo entre dos vectores.
        
        Args:
            vector1, vector2: Diccionarios con componentes x, y (y opcionalmente z)
            
        Returns:
            float: Ángulo en grados
        """
        
        if 'z' in vector1 and 'z' in vector2:
            mag1 = (vector1['x']**2 + vector1['y']**2 + vector1['z']**2)**0.5
            mag2 = (vector2['x']**2 + vector2['y']**2 + vector2['z']**2)**0.5
            
            
            dot_product = (vector1['x'] * vector2['x'] + 
                          vector1['y'] * vector2['y'] + 
                          vector1['z'] * vector2['z'])
        else:
            
            mag1 = (vector1['x']**2 + vector1['y']**2)**0.5
            mag2 = (vector2['x']**2 + vector2['y']**2)**0.5
            
            
            dot_product = vector1['x'] * vector2['x'] + vector1['y'] * vector2['y']
        
        
        cos_angle = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
        
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def process_frame(self, frame, frame_idx=None):
        """
        Procesa un frame de video para detectar poses y extraer información.
        Diseñado para usarse con la función process_video_file de VideoCapture.
        
        Args:
            frame: Frame de video (imagen BGR)
            frame_idx: Índice del frame (opcional)
            
        Returns:
            tuple: (results_dict, processed_frame) donde results_dict contiene landmarks y ángulos,
                  y processed_frame es el frame con anotaciones visuales
        """
        
        results, annotated_frame = self.detect_pose(frame)
        
        
        landmarks_dict = self.extract_landmarks(results, normalize=True)
        
        
        if not landmarks_dict:
            return {"frame_idx": frame_idx, "landmarks": None, "angles": None}, annotated_frame
        
        
        angles = self.calculate_joint_angles(landmarks_dict)
        
        
        if angles:
            
            y_pos = 30
            for angle_name, angle_value in angles.items():
                cv2.putText(annotated_frame, f"{angle_name}: {angle_value:.1f}°", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30
        
        
        results_dict = {
            "frame_idx": frame_idx,
            "landmarks": landmarks_dict,
            "angles": angles
        }
        
        return results_dict, annotated_frame
    
    def save_landmarks_to_csv(self, landmarks_list, output_path):
        """
        Guarda una lista de landmarks en formato CSV.
        
        Args:
            landmarks_list: Lista de diccionarios con los landmarks extraídos
            output_path: Ruta donde guardar el archivo CSV
        """
        if not landmarks_list or not output_path:
            return False
        
        
        valid_landmarks = [item for item in landmarks_list if item["landmarks"] is not None]
        if not valid_landmarks:
            print("No hay landmarks válidos para guardar")
            return False
        
        
        rows = []
        for item in valid_landmarks:
            frame_idx = item.get("frame_idx", None)
            landmarks = item["landmarks"]
            angles = item.get("angles", {})
            
            
            row = {"frame_idx": frame_idx}
            
            
            for landmark_name, landmark_data in landmarks.items():
                for coord in ["x", "y", "z", "visibility"]:
                    row[f"{landmark_name}_{coord}"] = landmark_data[coord]
            
            
            if angles:
                for angle_name, angle_value in angles.items():
                    row[angle_name] = angle_value
            
            rows.append(row)
        
        
        df = pd.DataFrame(rows)
        
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Landmarks guardados en {output_path}")
        
        return True


if __name__ == "__main__":
    
    pose_detector = PoseDetector()
    
    
    cap = cv2.VideoCapture(0)
    
    
    landmarks_list = []
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error al leer de la webcam.")
                break
            
            
            results_dict, annotated_image = pose_detector.process_frame(image, len(landmarks_list))
            
            
            landmarks_list.append(results_dict)
            
            
            cv2.imshow('MediaPipe Pose', annotated_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        
        if len(landmarks_list) > 10:
            pose_detector.save_landmarks_to_csv(landmarks_list, "data/processed_data/webcam_test.csv") 