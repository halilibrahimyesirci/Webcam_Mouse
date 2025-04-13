import pyautogui
import numpy as np
from screeninfo import get_monitors

class MouseController:
    def __init__(self):
        self.last_x = 0
        self.last_y = 0
        self.is_controlling = False
        self.smoothing = 0.3
        self.velocity_history = []
        self.max_history = 5
        self.adaptive_smoothing = True
        self.min_smoothing = 0.1
        self.max_smoothing = 0.6
        self._initialize_screen_bounds()

    def _initialize_screen_bounds(self):
        """Initialize screen boundaries"""
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        
        # Set up PyAutoGUI safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    def move_mouse(self, x, y):
        """Move mouse to specified screen coordinates"""
        if not self.is_controlling:
            return
            
        import pyautogui
        pyautogui.FAILSAFE = False
        
        # Ekran koordinatlarını al
        screen_width, screen_height = pyautogui.size()
        target_x = int(x * screen_width)
        target_y = int(y * screen_height)
        
        # Önceki pozisyondan farkı hesapla
        dx = target_x - self.last_x
        dy = target_y - self.last_y
        
        # Non-linear ölçekleme uygula
        dx = self._apply_non_linear_scaling(dx)
        dy = self._apply_non_linear_scaling(dy)
        
        # Adaptif smoothing hesapla
        current_smoothing = self._calculate_adaptive_smoothing(dx, dy) if self.adaptive_smoothing else self.smoothing
        
        # Yeni pozisyonu hesapla
        new_x = int(self.last_x + dx * (1 - current_smoothing))
        new_y = int(self.last_y + dy * (1 - current_smoothing))
        
        # Ekran sınırları içinde tut
        new_x = max(0, min(screen_width - 1, new_x))
        new_y = max(0, min(screen_height - 1, new_y))
        
        # Fareyi hareket ettir
        pyautogui.moveTo(new_x, new_y)
        
        # Son pozisyonu güncelle
        self.last_x = new_x
        self.last_y = new_y

    def start_control(self):
        """Start mouse control"""
        self.is_controlling = True
        self.last_position = None

    def stop_control(self):
        """Stop mouse control"""
        self.is_controlling = False
        self.last_position = None

    def set_smoothing(self, value):
        """Set base smoothing factor"""
        self.smoothing = value

    def _calculate_adaptive_smoothing(self, dx, dy):
        """Hıza bağlı adaptif smoothing hesapla"""
        velocity = (dx**2 + dy**2)**0.5
        
        # Hız geçmişini güncelle
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)
            
        # Son hızların ortalaması
        avg_velocity = sum(self.velocity_history) / len(self.velocity_history)
        
        # Hıza bağlı smoothing - hız arttıkça smoothing azalır
        if avg_velocity > 100:  # Hızlı hareket
            return self.min_smoothing
        elif avg_velocity < 20:  # Yavaş, hassas hareket
            return self.max_smoothing
        else:  # Ara değerler için linear interpolasyon
            t = (avg_velocity - 20) / 80  # 20-100 arasını 0-1'e normalize et
            return self.max_smoothing * (1-t) + self.min_smoothing * t
            
    def _apply_non_linear_scaling(self, value):
        """Hassas hareketler için non-linear ölçekleme"""
        # Küçük değerler için daha hassas, büyük değerler için daha hızlı
        sign = 1 if value >= 0 else -1
        abs_val = abs(value)
        
        if abs_val < 0.2:  # Çok küçük hareketler
            scaled = abs_val * 0.5  # Daha hassas
        elif abs_val < 0.5:  # Orta hareketler
            scaled = 0.1 + (abs_val - 0.2) * 1.0  # Linear
        else:  # Büyük hareketler
            scaled = 0.4 + (abs_val - 0.5) * 1.5  # Daha hızlı
            
        return sign * scaled

    def click(self):
        """Perform mouse click"""
        if self.is_controlling:
            pyautogui.click()

    def double_click(self):
        """Perform double click"""
        if self.is_controlling:
            pyautogui.doubleClick()

    def right_click(self):
        """Perform right click"""
        if self.is_controlling:
            pyautogui.rightClick()