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
        self._initialize_screen_size()

    def _initialize_screen_size(self):
        """Get the primary monitor resolution"""
        primary_monitor = get_monitors()[0]
        self.screen_width = primary_monitor.width
        self.screen_height = primary_monitor.height

    def add_calibration_point(self, screen_point, gaze_point, depth):
        """Add a calibration point and corresponding eye position for a specific depth"""
        if depth in self.calibration_data:
            self.calibration_data[depth]['points'].append(screen_point)
            self.calibration_data[depth]['gaze_points'].append(gaze_point)

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

            # Polynomial features için gaze noktalarını hazırla
            gaze_features = np.column_stack([
                gaze_points,
                np.multiply(gaze_points[:, 0], gaze_points[:, 1]),
                np.square(gaze_points[:, 0]),
                np.square(gaze_points[:, 1])
            ])

            # Her derinlik için dönüşüm matrisini hesapla
            try:
                self.transformation_matrices[depth] = np.linalg.lstsq(
                    gaze_features,
                    points,
                    rcond=None
                )[0]
            except np.linalg.LinAlgError:
                success = False
                continue

        return success

    def get_screen_position(self, gaze_point):
        """Convert eye position to screen coordinates using the appropriate transformation matrix"""
        if self.current_depth not in self.transformation_matrices or \
           self.transformation_matrices[self.current_depth] is None:
            return None

        # Create feature vector with polynomial features
        gaze_features = np.array([
            gaze_point[0],
            gaze_point[1],
            gaze_point[0] * gaze_point[1],
            gaze_point[0]**2,
            gaze_point[1]**2
        ])

        # Calculate screen position using current depth's transformation matrix
        screen_pos = np.dot(gaze_features, self.transformation_matrices[self.current_depth])
        
        # Ensure coordinates are within screen bounds
        x = np.clip(screen_pos[0], 0, self.screen_width)
        y = np.clip(screen_pos[1], 0, self.screen_height)
        
        return (int(x), int(y))

    def get_calibration_points(self):
        """Generate calibration points at screen corners and center"""
        padding = 50  # Pixels from screen edge
        points = [
            (padding, padding),  # Sol üst
            (self.screen_width - padding, padding),  # Sağ üst
            (self.screen_width//2, self.screen_height//2),  # Merkez
            (padding, self.screen_height - padding),  # Sol alt
            (self.screen_width - padding, self.screen_height - padding)  # Sağ alt
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
        if not self.calibration_data:
            return None

        all_screen_points = []
        all_gaze_points = []

        for point_data in self.calibration_data.values():
            target = point_data['target']
            gaze_points = point_data['gaze_points']
            
            if not gaze_points:
                continue
                
            avg_gaze = np.mean(gaze_points, axis=0)
            all_screen_points.append(target)
            all_gaze_points.append(avg_gaze)

        if len(all_screen_points) < 4:  # en az 4 nokta gerekli
            return None

        # Polynomial regression için hazırlık
        screen_points = np.array(all_screen_points)
        gaze_points = np.array(all_gaze_points)

        # X koordinatları için polinom katsayıları
        px = np.polyfit(gaze_points[:, 0], screen_points[:, 0], 2)
        # Y koordinatları için polinom katsayıları
        py = np.polyfit(gaze_points[:, 1], screen_points[:, 1], 2)

        return {'px': px, 'py': py}

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