import cv2
import mediapipe as mp
import numpy as np
import logging

# Configure logging to suppress TensorFlow warnings
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.camera = None
        self.camera_index = 0
        self.frame_width = None
        self.frame_height = None

        # MediaPipe iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Göz çevresi landmarkları - daha hassas ölçüm için genişletilmiş
        self.LEFT_EYE = [
            # Üst göz kapağı
            386, 374, 373, 390, 388, 387, 
            # Alt göz kapağı
            263, 249, 390, 373, 374, 380,
            # İç göz kenarı
            362, 382, 381, 380, 
            # Dış göz kenarı
            398, 384, 385, 386, 
            # Göz çukuru
            466, 388, 387, 386, 385, 384, 398
        ]
        
        self.RIGHT_EYE = [
            # Üst göz kapağı
            159, 145, 144, 163, 161, 160, 
            # Alt göz kapağı
            33, 7, 163, 144, 145, 153,
            # İç göz kenarı
            133, 155, 154, 153, 
            # Dış göz kenarı
            173, 157, 158, 159,
            # Göz çukuru
            246, 161, 160, 159, 158, 157, 173
        ]

        # Gürültü filtreleme için
        self.last_positions = []
        self.max_positions = 5  # Son 5 pozisyonu tut
        self.position_threshold = 0.1  # Ani değişim eşiği

    def _calculate_eye_center(self, landmarks, eye_points):
        """Göz merkezini hesapla - tüm göz noktalarının ortalaması"""
        eye_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                             for idx in eye_points])
        return np.mean(eye_coords, axis=0)

    def _calculate_iris_center(self, landmarks, iris_points):
        """Göz bebeği merkezini hesapla"""
        iris_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                              for idx in iris_points])
        return np.mean(iris_coords, axis=0)

    def _calculate_relative_iris_position(self, landmarks, iris_points, eye_points):
        """Göz bebeğinin göz içindeki relatif pozisyonunu hesapla"""
        # Göz çukuru sınırlarını bul
        eye_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                             for idx in eye_points])
        min_x, min_y = np.min(eye_coords, axis=0)
        max_x, max_y = np.max(eye_coords, axis=0)
        
        # Göz bebeği merkezini bul
        iris_center = self._calculate_iris_center(landmarks, iris_points)
        
        # Relatif pozisyonu hesapla (0-1 aralığında)
        rel_x = (iris_center[0] - min_x) / (max_x - min_x)
        rel_y = (iris_center[1] - min_y) / (max_y - min_y)
        
        # Göz kenarlarındaki anomalileri düzelt
        rel_x = np.clip(rel_x, 0.1, 0.9)  # Göz kenarlarında %10 tolerans
        rel_y = np.clip(rel_y, 0.1, 0.9)
        
        return np.array([rel_x, rel_y])

    def _estimate_depth(self, face_landmarks):
        """Derinlik tahmini - yüz genişliği ve göz mesafesini kullanarak"""
        # Yüz genişliği
        left_face = face_landmarks.landmark[234]
        right_face = face_landmarks.landmark[454]
        face_width = abs(right_face.x - left_face.x)
        
        # Gözler arası mesafe
        left_eye = self._calculate_eye_center(face_landmarks, self.LEFT_EYE)
        right_eye = self._calculate_eye_center(face_landmarks, self.RIGHT_EYE)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Normalize et (yaklaşık 0-1 aralığı)
        normalized_width = np.clip((face_width - 0.15) / (0.45 - 0.15), 0, 1)
        normalized_distance = np.clip((eye_distance - 0.05) / (0.15 - 0.05), 0, 1)
        
        # İki ölçümün ağırlıklı ortalaması
        depth = 0.7 * normalized_width + 0.3 * normalized_distance
        return depth

    def get_eye_position(self, frame):
        """Process frame and return eye position"""
        if frame is None:
            return None

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        
        # Her iki göz için relatif pozisyonları hesapla
        left_rel_pos = self._calculate_relative_iris_position(
            face_landmarks, self.LEFT_IRIS, self.LEFT_EYE)
        right_rel_pos = self._calculate_relative_iris_position(
            face_landmarks, self.RIGHT_IRIS, self.RIGHT_EYE)
        
        # Göz açıklığını kontrol et
        left_eye_open = self._calculate_eye_openness(face_landmarks, self.LEFT_EYE)
        right_eye_open = self._calculate_eye_openness(face_landmarks, self.RIGHT_EYE)
        
        # Göz açıklık eşiği
        eye_threshold = 0.15
        
        # Açıklık durumuna göre ağırlıkları belirle
        left_weight = 1.0 if left_eye_open > eye_threshold else 0.0
        right_weight = 1.0 if right_eye_open > eye_threshold else 0.0
        
        # Eğer iki göz de kapalıysa None döndür
        if left_weight == 0.0 and right_weight == 0.0:
            return None
            
        # Ağırlıklı ortalama al
        total_weight = left_weight + right_weight
        if total_weight > 0:
            weighted_pos = (
                left_rel_pos * left_weight + 
                right_rel_pos * right_weight
            ) / total_weight
        else:
            weighted_pos = (left_rel_pos + right_rel_pos) / 2.0
            
        # Pozisyonu yumuşat
        smoothed_pos = self._smooth_position(weighted_pos)
        
        # Get visualization points
        left_iris = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                             for idx in self.LEFT_IRIS])
        right_iris = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                              for idx in self.RIGHT_IRIS])
        
        depth = self._estimate_depth(face_landmarks)
        
        return {
            'left_eye': left_rel_pos,
            'right_eye': right_rel_pos,
            'gaze_point': smoothed_pos,
            'left_iris_points': left_iris,
            'right_iris_points': right_iris,
            'depth': depth,
            'original_frame': frame
        }

    def _smooth_position(self, new_position):
        """Pozisyon değişimlerini yumuşat"""
        if not self.last_positions:
            self.last_positions.append(new_position)
            return new_position
            
        # Ani değişimleri kontrol et
        last_pos = self.last_positions[-1]
        distance = np.linalg.norm(new_position - last_pos)
        
        if distance > self.position_threshold:
            # Ani değişim varsa, değişimi yumuşat
            smoothed_pos = last_pos + (new_position - last_pos) * 0.5
        else:
            smoothed_pos = new_position
            
        # Son pozisyonları güncelle
        self.last_positions.append(smoothed_pos)
        if len(self.last_positions) > self.max_positions:
            self.last_positions.pop(0)
            
        # Son pozisyonların ortalamasını al
        return np.mean(self.last_positions, axis=0)

    def _calculate_eye_openness(self, landmarks, eye_points):
        """Gözün ne kadar açık olduğunu hesapla"""
        eye_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                             for idx in eye_points])
        
        # Dikey mesafeyi hesapla
        min_y = np.min(eye_coords[:, 1])
        max_y = np.max(eye_coords[:, 1])
        vertical_distance = max_y - min_y
        
        # Yatay mesafeyi hesapla
        min_x = np.min(eye_coords[:, 0])
        max_x = np.max(eye_coords[:, 0])
        horizontal_distance = max_x - min_x
        
        # Göz açıklık oranı
        return vertical_distance / horizontal_distance if horizontal_distance > 0 else 0

    def start_camera(self, camera_index=0):
        """Initialize and start the webcam"""
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)
        
        if self.camera.isOpened():
            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return True
        return False

    def stop_camera(self):
        """Stop and release the webcam"""
        if self.camera is not None:
            self.camera.release()