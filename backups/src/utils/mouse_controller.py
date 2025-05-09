import pyautogui
import numpy as np

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class MouseController:
    def __init__(self):
        self.screen = pyautogui.size()
        self.is_controlling = False
        self.last_position = None
        self.smoothing_factor = 0.4  # Increased for smoother movement
        self.momentum_factor = 0.3  # Reduced for more responsive control
        self.vertical_sensitivity = 1.0
        self.velocity = np.array([0.0, 0.0])
        
        # Hareket parametreleri
        self.min_movement = 1  # Reduced minimum movement threshold
        self.max_speed = 80  # Reduced max speed for better control
        self.center_deadzone = 0.03  # Reduced dead zone
        self.edge_sensitivity = 1.2  # Reduced edge sensitivity
        self.screen_margin = 20  # Pixels from screen edge to stop at
        
        # Stabilizasyon için
        self.position_buffer = []
        self.buffer_size = 7  # Increased buffer size
        self.outlier_threshold = 1.5  # Reduced outlier threshold
        
        # Adaptif hassasiyet için
        self.base_sensitivity = 0.8  # Reduced base sensitivity
        self.speed_curve = 0.5  # More linear response
        self.center_weight = 0.8  # Increased center weight
        
    def set_smoothing(self, value):
        """Set mouse movement smoothing factor"""
        self.smoothing_factor = np.clip(value, 0.0, 1.0)
        
    def set_momentum(self, value):
        """Set mouse movement momentum factor"""
        self.momentum_factor = np.clip(value, 0.0, 1.0)
        
    def set_vertical_sensitivity(self, value):
        """Set vertical movement sensitivity"""
        self.vertical_sensitivity = value

    def _apply_speed_curve(self, movement):
        """Hareket hızını eğriye göre ayarla"""
        norm = np.linalg.norm(movement)
        if norm < self.min_movement:
            return np.zeros_like(movement)
            
        # Mesafeye bağlı hız eğrisi
        speed_factor = np.power(norm / self.max_speed, self.speed_curve)
        speed_factor = np.clip(speed_factor, 0.0, 1.0)
        
        # Normalize ve hız uygula
        return movement * speed_factor / norm if norm > 0 else movement

    def _calculate_adaptive_sensitivity(self, pos):
        """Pozisyona göre adaptif hassasiyet hesapla"""
        # Ekran merkezine olan uzaklık (normalize edilmiş)
        center_dist = np.array([0.5, 0.5]) - pos
        dist_norm = np.linalg.norm(center_dist)
        
        # Merkeze yakınken hassasiyeti düşür
        if dist_norm < self.center_deadzone:
            return self.base_sensitivity * (dist_norm / self.center_deadzone)
            
        # Kenarlara yakınken hassasiyeti artır
        if dist_norm > 0.4:  # Ekranın %40'ından uzakta
            edge_factor = np.clip((dist_norm - 0.4) / 0.1, 0, 1)
            return self.base_sensitivity * (1 + edge_factor * (self.edge_sensitivity - 1))
            
        return self.base_sensitivity

    def _stabilize_position(self, pos):
        """Pozisyonu stabilize et"""
        self.position_buffer.append(pos)
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
            
        if len(self.position_buffer) < 3:
            return pos
            
        # Medyan filtresi
        positions = np.array(self.position_buffer)
        median_pos = np.median(positions, axis=0)
        
        # Aykırı değerleri tespit et
        distances = np.linalg.norm(positions - median_pos, axis=1)
        std_dev = np.std(distances)
        
        # Aykırı değerleri filtrele
        valid_positions = positions[distances < self.outlier_threshold * std_dev]
        if len(valid_positions) > 0:
            return np.mean(valid_positions, axis=0)
        return median_pos

    def move_mouse(self, x, y, depth=None):
        """Move mouse based on gaze position"""
        if not self.is_controlling:
            return
            
        # Convert to numpy array for calculations
        current_pos = np.array([x, y])
        
        # Pozisyonu stabilize et
        stable_pos = self._stabilize_position(current_pos)
        
        # Adaptif hassasiyet uygula
        sensitivity = self._calculate_adaptive_sensitivity(stable_pos)
        
        if self.last_position is not None:
            # Hareket vektörünü hesapla
            movement = stable_pos - self.last_position
            
            # Hız eğrisini uygula
            curved_movement = self._apply_speed_curve(movement)
            
            # Momentum ve smoothing uygula
            self.velocity = (self.velocity * self.momentum_factor + 
                           curved_movement * (1 - self.momentum_factor))
            
            # Vertical speed adjustment
            self.velocity[1] *= self.vertical_sensitivity
            
            # Son pozisyonu güncelle
            target_pos = self.last_position + self.velocity * sensitivity
        else:
            target_pos = stable_pos
            
        self.last_position = stable_pos
        
        # Ekran koordinatlarına dönüştür
        screen_x = int(target_pos[0] * self.screen[0])
        screen_y = int(target_pos[1] * self.screen[1])
        
        # Sınırları kontrol et (ekran kenarlarından margin kadar uzak dur)
        screen_x = np.clip(screen_x, self.screen_margin, self.screen[0] - self.screen_margin - 1)
        screen_y = np.clip(screen_y, self.screen_margin, self.screen[1] - self.screen_margin - 1)
        
        # Fareyi hareket ettir
        pyautogui.moveTo(screen_x, screen_y)
        
    def start_control(self):
        """Enable mouse control"""
        self.is_controlling = True
        self.last_position = None
        self.velocity = np.array([0.0, 0.0])
        self.position_buffer = []
        
    def stop_control(self):
        """Disable mouse control"""
        self.is_controlling = False
        self.last_position = None