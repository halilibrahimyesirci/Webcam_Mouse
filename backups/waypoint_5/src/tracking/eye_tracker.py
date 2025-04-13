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

        # Iris detection için ek parametreler
        self.min_iris_radius = 0.01  # Minimum iris yarıçapı (normalize edilmiş)
        self.max_iris_radius = 0.04  # Maximum iris yarıçapı (normalize edilmiş)
        
        # Göz açıklığı için threshold değerleri
        self.min_eye_aspect_ratio = 0.15
        self.max_eye_aspect_ratio = 0.35

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
        """Göz bebeğinin göz içindeki pozisyonunu hassas şekilde hesapla"""
        try:
            # Göz çukuru sınırlarını bul
            eye_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                                for idx in eye_points])
            min_x, min_y = np.min(eye_coords, axis=0)
            max_x, max_y = np.max(eye_coords, axis=0)
            
            # Göz merkezi
            eye_center = np.mean(eye_coords, axis=0)
            
            # İris detaylarını al
            iris_details = self._calculate_iris_details(landmarks, iris_points)
            iris_center = iris_details['center']
            
            # Göz genişliği ve yüksekliği
            eye_width = max_x - min_x
            eye_height = max_y - min_y
            
            if eye_width == 0 or eye_height == 0:
                raise ValueError("Invalid eye dimensions")
            
            # İris'in göz merkezine göre pozisyonu (vektör)
            iris_vector = iris_center - eye_center
            
            # Normalize edilmiş pozisyon (-1 ile 1 arasında)
            rel_x = 2 * (iris_vector[0] / eye_width)
            rel_y = 2 * (iris_vector[1] / eye_height)
            
            # Sınırları kontrol et
            rel_x = np.clip(rel_x, -1, 1)
            rel_y = np.clip(rel_y, -1, 1)
            
            # İris boyut analizi
            iris_size = iris_details['radius'] * 2
            normalized_size = np.clip(
                (iris_size - self.min_iris_radius) / 
                (self.max_iris_radius - self.min_iris_radius),
                0, 1
            )
            
            # Göz açıklık oranı
            eye_aspect_ratio = np.clip(
                eye_height / eye_width if eye_width > 0 else 0,
                self.min_eye_aspect_ratio,
                self.max_eye_aspect_ratio
            )
            normalized_ear = (eye_aspect_ratio - self.min_eye_aspect_ratio) / \
                           (self.max_eye_aspect_ratio - self.min_eye_aspect_ratio)
            
            return np.array([rel_x, rel_y]), {
                'normalized_size': normalized_size,
                'aspect_ratio': normalized_ear,
                'regularity': iris_details['regularity'],
                'angle': iris_details['angle']
            }
            
        except Exception as e:
            # Hata durumunda varsayılan değerler
            return np.array([0, 0]), {
                'normalized_size': 0.5,
                'aspect_ratio': 0.5,
                'regularity': 1.0,
                'angle': 0
            }

    def _calculate_iris_details(self, landmarks, iris_points):
        """Göz bebeğinin detaylı analizini yap"""
        # İris noktalarını al
        iris_landmarks = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                                for idx in iris_points])
        
        # İris merkezi
        center = np.mean(iris_landmarks, axis=0)
        
        # İris yarıçapı - en uzak noktaların ortalaması
        distances = np.linalg.norm(iris_landmarks - center, axis=1)
        radius = np.mean(distances)

        # Elips parametrelerini hesapla
        if len(iris_landmarks) >= 4:
            # Kovaryans matrisi kullanarak yönelim hesapla
            points_centered = iris_landmarks - center
            cov = np.cov(points_centered.T)
            
            if cov.shape == (2, 2):  # Geçerli kovaryans matrisi kontrolü
                eigenvals, eigenvects = np.linalg.eig(cov)
                
                # En büyük özdeğere karşılık gelen vektör yönelimi verir
                major_idx = np.argmax(eigenvals)
                major_axis = eigenvects[:, major_idx]
                
                # Açıyı hesapla (-90 ile 90 derece arası)
                angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
                
                # Elips eksenleri - özdeğerlerden hesapla
                axes = (np.sqrt(eigenvals[major_idx]) * 3,  # Major axis
                       np.sqrt(eigenvals[1-major_idx]) * 3) # Minor axis
                
                # Elipsin düzenliliği
                regularity = min(axes) / max(axes) if max(axes) > 0 else 1.0
                
                return {
                    'center': center,
                    'radius': radius,
                    'angle': angle % 180,  # 0-180 arasına normalize et
                    'axes': axes,
                    'regularity': regularity
                }
        
        # Yeterli veri yoksa veya hesaplama başarısız olursa dairesel varsay
        return {
            'center': center,
            'radius': radius,
            'angle': 0,
            'axes': (radius, radius),
            'regularity': 1.0
        }

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
        
        # Her iki göz için detaylı pozisyon ve metrikleri hesapla
        left_pos, left_metrics = self._calculate_relative_iris_position(
            face_landmarks, self.LEFT_IRIS, self.LEFT_EYE)
        right_pos, right_metrics = self._calculate_relative_iris_position(
            face_landmarks, self.RIGHT_IRIS, self.RIGHT_EYE)
        
        # Göz açıklık kontrolü ve ağırlıklı pozisyon hesaplama
        left_weight = left_metrics['aspect_ratio'] * left_metrics['regularity']
        right_weight = right_metrics['aspect_ratio'] * right_metrics['regularity']
        
        total_weight = left_weight + right_weight
        if total_weight > 0:
            weighted_pos = (left_pos * left_weight + right_pos * right_weight) / total_weight
        else:
            weighted_pos = (left_pos + right_pos) / 2.0
            
        # Pozisyonu yumuşat
        smoothed_pos = self._smooth_position(weighted_pos)
        
        # Visualization için iris noktaları
        left_iris = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                             for idx in self.LEFT_IRIS])
        right_iris = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                              for idx in self.RIGHT_IRIS])
        
        depth = self._estimate_depth(face_landmarks)
        
        return {
            'left_eye': left_pos,
            'right_eye': right_pos,
            'gaze_point': smoothed_pos,
            'left_iris_points': left_iris,
            'right_iris_points': right_iris,
            'left_metrics': left_metrics,
            'right_metrics': right_metrics,
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