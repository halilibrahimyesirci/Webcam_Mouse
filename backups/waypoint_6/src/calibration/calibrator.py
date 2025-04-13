import numpy as np
from screeninfo import get_monitors
from time import time
import logging

class Calibrator:
    def __init__(self):
        # Her mesafe için ayrı kalibrasyon verileri
        self.calibration_data = {
            0: {'points': [], 'gaze_points': []},  # Yakın
            1: {'points': [], 'gaze_points': []},  # Orta
            2: {'points': [], 'gaze_points': []}   # Uzak
        }
        self.transformation_matrices = {
            0: None,  # Yakın mesafe için dönüşüm matrisi
            1: None,  # Orta mesafe için dönüşüm matrisi
            2: None   # Uzak mesafe için dönüşüm matrisi
        }
        self.current_depth = 1  # Varsayılan olarak orta mesafe
        self.screen_width = None
        self.screen_height = None
        self.version = 2  # Kalibrasyon versiyon numarası
        self._initialize_screen_size()

    def _initialize_screen_size(self):
        """Get the primary monitor resolution"""
        primary_monitor = get_monitors()[0]
        self.screen_width = primary_monitor.width
        self.screen_height = primary_monitor.height

    def _create_feature_vector(self, gaze_point, version=None):
        """Versiyon uyumlu özellik vektörü oluştur"""
        version = version or self.version
        
        if version == 1:  # Eski versiyon (5 boyutlu)
            return np.array([
                1.0,  # Bias
                gaze_point[0], gaze_point[1],  # Linear
                gaze_point[0] * gaze_point[1],  # Cross
                gaze_point[0]**2  # Single quadratic
            ])
        else:  # Yeni versiyon (10 boyutlu)
            return np.array([
                1.0,  # Bias
                gaze_point[0], gaze_point[1],  # Linear
                gaze_point[0] * gaze_point[1],  # Cross
                gaze_point[0]**2, gaze_point[1]**2,  # Quadratic
                gaze_point[0]**3, gaze_point[1]**3,  # Cubic
                gaze_point[0]**2 * gaze_point[1],  # Mixed
                gaze_point[0] * gaze_point[1]**2  # Mixed
            ])

    def _detect_matrix_version(self, matrix):
        """Dönüşüm matrisinin versiyonunu tespit et"""
        return 1 if matrix.shape[0] == 5 else 2

    def get_screen_position(self, gaze_point):
        """Convert eye position to screen coordinates using the appropriate transformation matrix"""
        if self.current_depth not in self.transformation_matrices or \
           self.transformation_matrices[self.current_depth] is None:
            return None

        matrix = self.transformation_matrices[self.current_depth]
        matrix_version = self._detect_matrix_version(matrix)
        
        # Versiyon uyumlu özellik vektörü oluştur
        gaze_features = self._create_feature_vector(gaze_point, version=matrix_version)

        # Calculate screen position using current depth's transformation matrix
        screen_pos = np.dot(gaze_features, matrix)
        
        # Ensure coordinates are within screen bounds
        x = np.clip(screen_pos[0], 0, self.screen_width)
        y = np.clip(screen_pos[1], 0, self.screen_height)
        
        return (x, y)

    def calculate_transformation(self):
        """Calculate transformation matrices for all depths"""
        success = True
        
        for depth in self.calibration_data:
            points = self.calibration_data[depth]['points']
            gaze_points = self.calibration_data[depth]['gaze_points']
            
            if len(points) < 4:  # Her derinlik için en az 4 nokta gerekli
                success = False
                continue

            # Numpy array'e dönüştür
            gaze_points = np.array(gaze_points)
            points = np.array(points)

            # Özellik matrisini hazırla
            features = []
            for gaze_point in gaze_points:
                features.append(self._create_feature_vector(gaze_point))
            gaze_features = np.array(features)

            # SVD kullanarak daha stabil bir çözüm elde et
            try:
                # Moore-Penrose pseudoinverse kullanarak dönüşüm matrisini hesapla
                self.transformation_matrices[depth] = np.linalg.pinv(gaze_features) @ points
            except np.linalg.LinAlgError:
                success = False
                continue

        return success

    def add_calibration_point(self, screen_point, gaze_point, depth):
        """Add a calibration point and corresponding eye position for a specific depth"""
        if depth in self.calibration_data:
            self.calibration_data[depth]['points'].append(screen_point)
            self.calibration_data[depth]['gaze_points'].append(gaze_point)

    def get_calibration_points(self):
        """Generate calibration points in a 3x3 grid"""
        padding = 50  # Pixels from screen edge
        w = self.screen_width
        h = self.screen_height
        
        # 3x3 grid oluştur
        points = [
            # Üst satır
            (padding, padding),                    # Sol üst
            (w//2, padding),                       # Orta üst 
            (w - padding, padding),                # Sağ üst
            
            # Orta satır
            (padding, h//2),                       # Sol orta
            (w//2, h//2),                          # Merkez
            (w - padding, h//2),                   # Sağ orta
            
            # Alt satır
            (padding, h - padding),                # Sol alt
            (w//2, h - padding),                   # Orta alt
            (w - padding, h - padding)             # Sağ alt
        ]
        return points

    def set_depth(self, depth):
        """Set current depth for transformation selection"""
        if depth in self.transformation_matrices and self.transformation_matrices[depth] is not None:
            self.current_depth = depth
            return True
        return False

    def reset(self):
        """Reset calibration data"""
        for depth in self.calibration_data:
            self.calibration_data[depth] = {'points': [], 'gaze_points': []}
            self.transformation_matrices[depth] = None

    def start_calibration(self):
        """Kalibrasyonu başlat"""
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # üst sıra
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # orta sıra
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # alt sıra
        ]
        self.current_gaze_points = []
        self.calibration_data = {}
        self.is_calibrating = True
        self.current_point_index = 0
        self.point_start_time = time()
        self.last_stable_points = []

    def _check_point_stability(self, gaze_points):
        """Son gaze noktalarının stabilitesini kontrol et"""
        if len(gaze_points) < 5:
            return False
            
        recent_points = np.array(gaze_points[-5:])
        x_std = np.std(recent_points[:, 0])
        y_std = np.std(recent_points[:, 1])
        
        return x_std < self.stability_threshold and y_std < self.stability_threshold

    def _filter_outliers(self, points):
        """Aykırı değerleri temizle"""
        if len(points) < 4:
            return points
            
        points_array = np.array(points)
        mean = np.mean(points_array, axis=0)
        std = np.std(points_array, axis=0)
        
        filtered_points = []
        for point in points:
            if (abs(point[0] - mean[0]) < 2 * std[0] and 
                abs(point[1] - mean[1]) < 2 * std[1]):
                filtered_points.append(point)
                
        return filtered_points

    def add_gaze_point(self, gaze_point):
        """Göz bakış noktasını ekle ve kalibrasyon durumunu güncelle"""
        if not self.is_calibrating:
            return None

        current_time = time()
        elapsed_time = current_time - self.point_start_time

        # Geçiş süresi kontrolü
        if elapsed_time < self.transition_delay:
            return self.get_current_point()

        self.current_gaze_points.append(gaze_point)
        
        # Stabilite kontrolü
        is_stable = self._check_point_stability(self.current_gaze_points)
        
        if is_stable:
            self.last_stable_points.append(gaze_point)
        else:
            self.last_stable_points = []

        # Yeterli veri toplandı mı kontrolü
        enough_stable_points = len(self.last_stable_points) >= self.min_samples_per_point
        timeout_reached = elapsed_time >= self.point_timeout
        max_samples_reached = len(self.current_gaze_points) >= self.max_samples_per_point

        if (enough_stable_points or timeout_reached or max_samples_reached):
            # Noktayı kaydet ve sonrakine geç
            points_to_save = self.last_stable_points if enough_stable_points else self.current_gaze_points
            filtered_points = self._filter_outliers(points_to_save)
            
            if filtered_points:
                self.calibration_data[self.current_point_index] = {
                    'target': self.calibration_points[self.current_point_index],
                    'gaze_points': filtered_points
                }
            
            self.current_point_index += 1
            self.current_gaze_points = []
            self.last_stable_points = []
            self.point_start_time = current_time

            if self.current_point_index >= len(self.calibration_points):
                self.is_calibrating = False
                return None

        return self.get_current_point()

    def get_current_point(self):
        """Mevcut kalibrasyon noktasını döndür"""
        if not self.is_calibrating or self.current_point_index >= len(self.calibration_points):
            return None
        return self.calibration_points[self.current_point_index]

    def calculate_mapping(self):
        """Kalibrasyon verilerinden haritalama hesapla"""
        success = True
        
        for depth in self.calibration_data:
            points = self.calibration_data[depth]['points']
            gaze_points = self.calibration_data[depth]['gaze_points']
            
            if len(points) < 6:  # En az 6 nokta gerekli
                success = False
                continue
                
            gaze_array = np.array(gaze_points)
            points_array = np.array(points)
            
            # 3D göz hareketi analizi
            gaze_vectors = []
            screen_points = []
            
            for gaze, target in zip(gaze_points, points):
                # Göz açıları ve pozisyondan 3D vektör oluştur
                theta = np.arctan2(gaze[1], gaze[0])  # Yatay açı
                phi = np.arcsin(np.clip(gaze[1], -1, 1))  # Dikey açı
                
                gaze_vector = np.array([
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi),
                    np.cos(phi) * np.cos(theta)
                ])
                
                gaze_vectors.append(gaze_vector)
                screen_points.append(target)
            
            try:
                # Geometrik dönüşüm matrici
                geom_transform = self._calculate_geometric_transform(
                    np.array(gaze_vectors)[:, :2],  # XY projeksiyon
                    np.array(screen_points)
                )
                
                if geom_transform is not None:
                    self.transformation_matrices[depth] = geom_transform
                else:
                    # Geometrik dönüşüm başarısız olursa polynomial regresyona geri dön
                    px = np.polyfit(gaze_array[:, 0], points_array[:, 0], 2)
                    py = np.polyfit(gaze_array[:, 1], points_array[:, 1], 2)
                    self.transformation_matrices[depth] = {'px': px, 'py': py}
                    
            except np.linalg.LinAlgError:
                success = False
                continue
                
        return success

    def map_gaze_to_screen(self, gaze_point, mapping):
        """Göz bakışını ekran koordinatlarına dönüştür"""
        if mapping is None:
            return gaze_point

        px = mapping['px']
        py = mapping['py']

        screen_x = np.polyval(px, gaze_point[0])
        screen_y = np.polyval(py, gaze_point[1])

        # Değerleri 0-1 aralığında sınırla
        screen_x = max(0, min(1, screen_x))
        screen_y = max(0, min(1, screen_y))

        return (screen_x, screen_y)

    def _project_gaze_to_screen(self, gaze_vector, head_position):
        """3D bakış vektörünü ekran düzlemine projeksiyon yap"""
        # Ekranın yaklaşık pozisyonu (kameradan)
        screen_normal = np.array([0, 0, 1])  # Ekran düzlemi normali
        screen_distance = 0.6  # Metre cinsinden yaklaşık ekran mesafesi
        
        # Gaze vektörünün ekranla kesişim noktasını bul
        # Düzlem-doğru kesişimi formülü
        d = np.dot(screen_normal, gaze_vector)
        if abs(d) < 1e-6:  # Bakış ekrana paralel
            return None
            
        # Kesişim noktasını hesapla
        t = (screen_distance - np.dot(screen_normal, head_position)) / d
        intersection = head_position + t * gaze_vector
        
        # Ekran koordinatlarına dönüştür
        screen_x = (intersection[0] / screen_distance + 1) / 2 * self.screen_width
        screen_y = (intersection[1] / screen_distance + 1) / 2 * self.screen_height
        
        return np.array([screen_x, screen_y])

    def _calculate_geometric_transform(self, gaze_points, screen_points):
        """Geometrik dönüşüm matrisini hesapla"""
        if len(gaze_points) < 4:
            return None
            
        # Homografik dönüşüm matrisi hesapla
        try:
            # Noktaları homogen koordinatlara dönüştür
            gaze_homog = np.column_stack([gaze_points, np.ones(len(gaze_points))])
            screen_homog = np.column_stack([screen_points, np.ones(len(screen_points))])
            
            # SVD ile çöz
            U, S, Vt = np.linalg.svd(gaze_homog)
            
            # Projeksiyon matrisini hesapla
            V = Vt.T
            H = V[:, :2] @ np.diag(S[:2]) @ U.T[:2, :] @ screen_homog
            
            return H
        except:
            return None