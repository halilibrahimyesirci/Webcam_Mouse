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
            min_detection_confidence=0.6,  # Arttırıldı
            min_tracking_confidence=0.6    # Arttırıldı
        )
        self.camera = None
        self.camera_index = 0
        self.frame_width = None
        self.frame_height = None

        # İris landmarks
        self.LEFT_IRIS = [474,475,476,477]
        self.RIGHT_IRIS = [469,470,471,472]
        
        # Genişletilmiş göz landmarkları aynen kalır
        self.LEFT_EYE = [
            # Üst göz kapağı detaylı
            246, 161, 160, 159, 158, 157, 173,  
            # Alt göz kapağı detaylı
            33, 7, 163, 144, 145, 153, 154, 155,
            # Göz köşeleri detaylı
            133, 173, 157, 158, 159, 160, 161, 246,
            # Göz çevresi konturları
            124, 46, 53, 52, 65, 55, 193,
            # İris referans noktaları
            469, 470, 471, 472
        ]
        
        self.RIGHT_EYE = [
            # Üst göz kapağı detaylı
            466, 388, 387, 386, 385, 384, 398,
            # Alt göz kapağı detaylı
            263, 249, 390, 373, 374, 380, 381, 382,
            # Göz köşeleri detaylı
            362, 398, 384, 385, 386, 387, 388, 466,
            # Göz çevresi konturları
            351, 275, 282, 281, 295, 285, 417,
            # İris referans noktaları
            474, 475, 476, 477
        ]

        # Filtreleme ve kalite kontrol parametreleri - iyileştirildi
        self.min_eye_aspect_ratio = 0.12  # Düşürüldü
        self.max_eye_aspect_ratio = 0.4   # Arttırıldı
        self.min_iris_size = 0.008        # Düşürüldü
        self.max_iris_size = 0.05         # Arttırıldı
        self.movement_threshold = 0.015    # Düşürüldü
        
        # Pozisyon smoothing için - iyileştirildi
        self.position_history = []
        self.max_history = 12             # Arttırıldı
        self.momentum = 0.65              # Düşürüldü
        self.velocity = np.zeros(2)
        
        # Kalite kontrol için - iyileştirildi
        self.min_detection_confidence = 0.8  # Arttırıldı
        self.outlier_threshold = 1.8        # Düşürüldü
        self.min_tracking_quality = 0.7      # Arttırıldı
        
        # Göz hareketi analizi için - iyileştirildi
        self.saccade_threshold = 0.08     # Düşürüldü
        self.fixation_threshold = 0.015    # Düşürüldü
        self.fixation_duration = 0.08      # Düşürüldü

        # Yeni - Kalman filtresi için
        self.kalman_filter_initialized = False
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, dx, dy), 2 measurement vars (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0],
                                              [0, 1e-4, 0, 0],
                                              [0, 0, 1e-4, 0],
                                              [0, 0, 0, 1e-4]], np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                   [0, 1]], np.float32) * 0.1

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
            
            # Üst ve alt göz kapağı noktalarını ayrı ayrı ele al
            top_points = eye_coords[:len(eye_points)//2]  # Üst yarı
            bottom_points = eye_coords[len(eye_points)//2:]  # Alt yarı
            
            # Göz kapakları için en uç noktaları bul
            top_y = np.min(top_points[:, 1])  # En üst nokta
            bottom_y = np.max(bottom_points[:, 1])  # En alt nokta
            min_x = np.min(eye_coords[:, 0])
            max_x = np.max(eye_coords[:, 0])
            
            # Göz merkezi - üst ve alt noktaların ortası
            eye_center_y = (top_y + bottom_y) / 2
            eye_center_x = (min_x + max_x) / 2
            eye_center = np.array([eye_center_x, eye_center_y])
            
            # İris detaylarını al
            iris_details = self._calculate_iris_details(landmarks, iris_points)
            iris_center = iris_details['center']
            
            # Göz genişliği ve yüksekliği
            eye_width = max_x - min_x
            eye_height = bottom_y - top_y  # Üst ve alt kapak arası mesafe
            
            if eye_width == 0 or eye_height == 0:
                raise ValueError("Invalid eye dimensions")
            
            # İris'in göz merkezine göre pozisyonu (vektör)
            iris_vector = iris_center - eye_center
            
            # Dikey hareket için özel ağırlık - göz açıklığına göre
            eye_aspect_ratio = eye_height / eye_width if eye_width > 0 else 1.0
            vertical_weight = np.clip(eye_aspect_ratio / 0.5, 0.5, 2.0)  # 0.5 ideal göz açıklık oranı
            
            # İris'in şekil düzenliliği - eliptik distorsiyonu kontrol et
            iris_shape_factor = iris_details['regularity']
            
            # Normalize edilmiş pozisyon (-1 ile 1 arasında)
            rel_x = 2 * (iris_vector[0] / eye_width) 
            rel_y = 2 * (iris_vector[1] / eye_height) * vertical_weight  # Dikey ağırlık uygula
            
            # İris boyut ve açı faktörlerini kullan
            iris_size_factor = iris_details['radius'] / (eye_height * 0.3)  # İdeal iris/göz oranı 0.3
            iris_angle_factor = np.cos(np.deg2rad(iris_details['angle']))  # Açı etkisini hesapla
            
            # Dinamik sınırlar - göz durumuna göre ayarla
            x_limit = 0.8 * iris_shape_factor  # Düzensiz iris şeklinde daha dar sınırlar
            y_limit = 0.8 * iris_shape_factor * vertical_weight  # Dikey limit göz açıklığına bağlı
            
            # Sınırları uygula
            rel_x = np.clip(rel_x * iris_size_factor * iris_angle_factor, -x_limit, x_limit)
            rel_y = np.clip(rel_y * iris_size_factor * iris_angle_factor, -y_limit, y_limit)
            
            # İris boyut analizi
            iris_size = iris_details['radius'] * 2
            normalized_size = np.clip(
                (iris_size - self.min_iris_size) / 
                (self.max_iris_size - self.min_iris_size),
                0, 1
            )
            
            # Göz açıklık oranını normalize et
            normalized_ear = np.clip(
                (eye_aspect_ratio - self.min_eye_aspect_ratio) / 
                (self.max_eye_aspect_ratio - self.min_eye_aspect_ratio),
                0, 1
            )
            
            return np.array([rel_x, rel_y]), {
                'normalized_size': normalized_size,
                'aspect_ratio': normalized_ear,
                'regularity': iris_details['regularity'],
                'angle': iris_details['angle'],
                'vertical_confidence': vertical_weight * iris_shape_factor  # Dikey takip güvenilirliği
            }
            
        except Exception as e:
            # Hata durumunda varsayılan değerler
            return np.array([0, 0]), {
                'normalized_size': 0.5,
                'aspect_ratio': 0.5,
                'regularity': 1.0,
                'angle': 0,
                'vertical_confidence': 0.5
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

    def _calculate_3d_gaze(self, landmarks, iris_points, eye_points):
        """3D göz pozisyonunu hesapla"""
        # İris elipsini bul
        iris_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y)
                              for idx in iris_points])
        
        # Göz çukuru sınırlarını bul
        eye_coords = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y)
                             for idx in eye_points])
        
        # Elips parametrelerini hesapla
        iris_center = np.mean(iris_coords, axis=0)
        centered_coords = iris_coords - iris_center
        
        # Kovaryans matrisi
        cov = np.cov(centered_coords.T)
        eigenvals, eigenvects = np.linalg.eig(cov)
        
        # Major ve minor eksenler
        major_axis = eigenvects[:, np.argmax(eigenvals)]
        minor_axis = eigenvects[:, np.argmin(eigenvals)]
        
        # Elips distorsiyonu - dairesel projeksiyondan sapma
        axis_ratio = np.sqrt(min(eigenvals) / max(eigenvals))
        
        # Göz küresinin tahmini yarıçapı (normalize edilmiş birimde)
        eye_width = np.max(eye_coords[:, 0]) - np.min(eye_coords[:, 0])
        eye_sphere_radius = eye_width * 1.2  # Göz genişliğinin ~1.2 katı
        
        # İris'in göz küresi üzerindeki konumunu hesapla
        # Elips distorsiyonundan 3D açıyı tahmin et
        gaze_angle_horizontal = np.arccos(axis_ratio)  # Radyan cinsinden
        
        # Dikey açı için iris pozisyonunu kullan
        eye_height = np.max(eye_coords[:, 1]) - np.min(eye_coords[:, 1])
        vertical_pos = (iris_center[1] - np.mean(eye_coords[:, 1])) / (eye_height/2)
        gaze_angle_vertical = np.arcsin(np.clip(vertical_pos, -1, 1))
        
        # 3D bakış vektörü oluştur
        gaze_vector = np.array([
            np.sin(gaze_angle_horizontal),
            np.sin(gaze_angle_vertical),
            -np.cos(gaze_angle_horizontal) * np.cos(gaze_angle_vertical)
        ])
        
        # Güven skoru hesapla
        confidence = axis_ratio * (1.0 - abs(vertical_pos))
        
        return gaze_vector, {
            'confidence': confidence,
            'iris_ellipse': {
                'center': iris_center,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'ratio': axis_ratio
            },
            'angles': {
                'horizontal': np.degrees(gaze_angle_horizontal),
                'vertical': np.degrees(gaze_angle_vertical)
            }
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

    def _draw_debug_visualization(self, frame, landmarks, left_pos, right_pos, left_metrics, right_metrics):
        """Debug görselleştirmesi çiz"""
        h, w = frame.shape[:2]
        
        # Landmark noktalarını çiz
        left_eye_points = []
        right_eye_points = []
        
        for idx in self.LEFT_EYE:
            pos = landmarks.landmark[idx]
            x, y = int(pos.x * w), int(pos.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)  # Sarı
            left_eye_points.append((x, y))
            
        for idx in self.RIGHT_EYE:
            pos = landmarks.landmark[idx]
            x, y = int(pos.x * w), int(pos.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)  # Sarı
            right_eye_points.append((x, y))
            
        # İris merkezlerini ve elipsleri çiz
        left_iris_center = None
        right_iris_center = None
        
        for idx in self.LEFT_IRIS:
            pos = landmarks.landmark[idx]
            x, y = int(pos.x * w), int(pos.y * h)
            if left_iris_center is None:  # İlk nokta merkez olsun
                left_iris_center = (x, y)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Mavi merkez
                
        for idx in self.RIGHT_IRIS:
            pos = landmarks.landmark[idx]
            x, y = int(pos.x * w), int(pos.y * h)
            if right_iris_center is None:  # İlk nokta merkez olsun
                right_iris_center = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Kırmızı merkez
        
        # Göz merkezlerini hesapla
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        
        # İris pozisyonlarını koordinat sistemine dönüştür
        left_scaled = (
            int(left_eye_center[0] + left_pos[0] * 30),  # X offset
            int(left_eye_center[1] + left_pos[1] * 30)   # Y offset
        )
        right_scaled = (
            int(right_eye_center[0] + right_pos[0] * 30),
            int(right_eye_center[1] + right_pos[1] * 30)
        )
        
        # Göz vektörlerini çiz
        cv2.arrowedLine(frame, tuple(left_eye_center), left_scaled, (255, 0, 0), 2)  # Mavi ok
        cv2.arrowedLine(frame, tuple(right_eye_center), right_scaled, (0, 0, 255), 2)  # Kırmızı ok
        
        # Güven metriklerini göster
        left_conf = f"L: {left_metrics['vertical_confidence']:.2f}"
        right_conf = f"R: {right_metrics['vertical_confidence']:.2f}"
        ear_text = f"EAR: {left_metrics['aspect_ratio']:.2f}/{right_metrics['aspect_ratio']:.2f}"
        
        y_offset = 30
        cv2.putText(frame, left_conf, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, right_conf, (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, ear_text, (10, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

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
        
        # Debug görselleştirmesi ekle
        debug_frame = frame.copy()
        debug_frame = self._draw_debug_visualization(
            debug_frame, face_landmarks, left_pos, right_pos, left_metrics, right_metrics)
        
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
            'debug_frame': debug_frame,
            'original_frame': frame
        }

    def _smooth_position(self, new_position):
        """Geliştirilmiş pozisyon yumuşatma"""
        if not self.position_history:
            self.position_history.append(new_position)
            self.kalman_filter_initialized = False
            return new_position

        # Kalman filtresi başlatma
        if not self.kalman_filter_initialized:
            self.kalman.statePost = np.array([[new_position[0]],
                                            [new_position[1]],
                                            [0],
                                            [0]], np.float32)
            self.kalman_filter_initialized = True
            return new_position

        # Kalman tahmini
        prediction = self.kalman.predict()
        measurement = np.array([[new_position[0]], [new_position[1]]], np.float32)
        corrected = self.kalman.correct(measurement)

        # Son pozisyondan farkı hesapla
        last_pos = self.position_history[-1]
        movement_vector = new_position - last_pos
        distance = np.linalg.norm(movement_vector)

        # Hareket hızına bağlı adaptif yumuşatma
        if distance > self.movement_threshold:
            # Hızlı hareket - daha az yumuşatma
            alpha = min(distance / 0.1, 0.8)  # Max 0.8 yumuşatma
            smoothed_pos = last_pos + (new_position - last_pos) * alpha
        else:
            # Yavaş hareket - Kalman ve momentum karışımı
            kalman_pos = np.array([corrected[0, 0], corrected[1, 0]])
            smoothed_pos = (kalman_pos * 0.7 + new_position * 0.3)

        # Aykırı değer kontrolü
        if len(self.position_history) >= 3:
            recent_positions = np.array(self.position_history[-3:])
            mean_pos = np.mean(recent_positions, axis=0)
            std_dev = np.std(recent_positions, axis=0)
            
            # Aykırı değer tespiti
            deviation = np.abs(smoothed_pos - mean_pos)
            if np.any(deviation > self.outlier_threshold * std_dev):
                # Aykırı değer bulundu - son geçerli pozisyona yakınsat
                smoothed_pos = mean_pos + (smoothed_pos - mean_pos) * 0.3

        # Geçmiş pozisyonları güncelle
        self.position_history.append(smoothed_pos)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        # Son pozisyonların ağırlıklı ortalaması
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weighted_average = np.average(self.position_history, weights=weights, axis=0)

        return weighted_average

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