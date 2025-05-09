import cv2
import numpy as np
import os
import dlib

class EyeTracker:
    def __init__(self, predictor_path):
        """
        Eye Tracker sınıfı başlatıcı
        
        Args:
            predictor_path: dlib'in yüz landmark dosyasının yolu
        """
        self.predictor = dlib.shape_predictor(predictor_path)
        self.initial_pupil_pos = None
        self.current_eye_roi = None
        self.eye_width = None
        self.eye_height = None
        self.base_eye_ratio = None
        self.is_left_eye = None
        # Göz takibi için gerekli parametreleri sıfırla
        self.reset_pupil_position()
        self.calibration = None
        
    def get_eye_landmarks(self, frame, face_rect_dlib):
        """Detects facial landmarks and extracts eye landmarks."""
        if frame is None or face_rect_dlib is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shapes = self.predictor(gray, face_rect_dlib)
        
        # Extract left and right eye landmarks
        left_eye_pts = np.array([(shapes.part(i).x, shapes.part(i).y) for i in range(36, 42)], dtype=np.int32)
        right_eye_pts = np.array([(shapes.part(i).x, shapes.part(i).y) for i in range(42, 48)], dtype=np.int32)
        
        return left_eye_pts, right_eye_pts

    def get_eye_bounding_box(self, landmarks):
        """Calculates a bounding box for a set of eye landmarks."""
        if landmarks is None or len(landmarks) == 0:
            return None
        x_min = np.min(landmarks[:, 0])
        x_max = np.max(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        y_max = np.max(landmarks[:, 1])
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def select_eye(self, frame, eye_landmarks_pts, is_left_eye):
        """Selects an eye based on landmarks and returns its ROI."""
        if frame is None or eye_landmarks_pts is None:
            self.selected_eye_landmarks = None
            self.eye_roi = None
            return None
            
        self.selected_eye_landmarks = eye_landmarks_pts

        # Get bounding box for the selected eye
        x, y, w, h = self.get_eye_bounding_box(eye_landmarks_pts)
        if w == 0 or h == 0:
             self.eye_roi = None
             return None

        # Add padding to eye ROI for better pupil tracking/display
        padding = 10
        x_roi = max(0, x - padding)
        y_roi = max(0, y - padding)
        w_roi = min(frame.shape[1] - x_roi, w + 2*padding)
        h_roi = min(frame.shape[0] - y_roi, h + 2*padding)
        
        self.eye_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
        return self.eye_roi

    def detect_pupil(self, eye_roi):
        """
        Göz ROI'sinde pupil tespiti yap
        
        Args:
            eye_roi: Göz bölgesi görüntüsü
        
        Returns:
            Pupil merkez koordinatları (x, y) veya None
        """
        if eye_roi is None or eye_roi.size == 0:
            print("Göz ROI boş, pupil tespiti yapılamıyor")
            return None
            
        # ROI'yi kaydet
        self.current_eye_roi = eye_roi.copy()
        
        # Görüntüyü gri tonlamaya dönüştür
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azalt ve kontrastı artır
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, thresh_eye = cv2.threshold(gray_eye, 35, 255, cv2.THRESH_BINARY_INV)
        
        # Konturları bul
        contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # Manuel olarak ROI merkezini kullan
            h, w = eye_roi.shape[:2]
            pupil_position = (w//2, h//2)
            cv2.circle(eye_roi, pupil_position, 3, (0, 255, 255), -1)
            
            # İlk pupil konumunu otomatik ayarla (eğer henüz ayarlanmamışsa)
            if self.initial_pupil_pos is None:
                self.initial_pupil_pos = pupil_position
                print(f"İlk pupil konumu otomatik ayarlandı: {self.initial_pupil_pos}")
                
            return pupil_position
        
        # En büyük konturu bul (muhtemelen pupil)
        pupil_contour = max(contours, key=cv2.contourArea)
        
        # Konturun merkezini hesapla
        M = cv2.moments(pupil_contour)
        if M["m00"] != 0:
            pupil_x = int(M["m10"] / M["m00"])
            pupil_y = int(M["m01"] / M["m00"])
            # Pupil'i çiz
            cv2.circle(eye_roi, (pupil_x, pupil_y), 3, (0, 255, 0), -1)
            
            # İlk pupil konumunu otomatik ayarla (eğer henüz ayarlanmamışsa)
            if self.initial_pupil_pos is None:
                self.initial_pupil_pos = (pupil_x, pupil_y)
                print(f"İlk pupil konumu otomatik ayarlandı: {self.initial_pupil_pos}")
                
            return (pupil_x, pupil_y)
        else:
            # ROI merkezini kullan
            h, w = eye_roi.shape[:2]
            pupil_position = (w//2, h//2)
            
            # İlk pupil konumunu otomatik ayarla (eğer henüz ayarlanmamışsa)
            if self.initial_pupil_pos is None:
                self.initial_pupil_pos = pupil_position
                print(f"İlk pupil konumu otomatik ayarlandı: {self.initial_pupil_pos}")
                
            return pupil_position

    def calculate_pupil_movement(self, pupil_pos, eye_height=None, eye_aspect_ratio=None):
        """
        Göz kapağı kapalılığına ve göz pozisyonuna göre hareket hesapla
        
        Args:
            pupil_pos: Gözbebeği pozisyonu (x, y)
            eye_height: Göz yüksekliği (üst-alt mesafesi)
            eye_aspect_ratio: Göz açıklık oranı (height/width)
        """
        print(f"calculate_pupil_movement çağrıldı: pupil_pos={pupil_pos}, eye_height={eye_height}, eye_ratio={eye_aspect_ratio}")
        
        if self.initial_pupil_pos is None:
            print("İlk pupil konumu ayarlanmamış!")
            # İlk pupil konumunu otomatik ayarla
            self.initial_pupil_pos = pupil_pos
            self.base_eye_ratio = eye_aspect_ratio if eye_aspect_ratio is not None else 0.25
            print(f"İlk pupil konumu otomatik ayarlandı: {self.initial_pupil_pos}")
            return (0, 0)  # Hareketsiz başlangıç noktası
            
        if pupil_pos is None:
            print("Pupil konumu algılanmadı!")
            return None
        
        # Göz ROI boyutlarını al
        if not hasattr(self, 'eye_width') or not hasattr(self, 'eye_height') or self.eye_width is None or self.eye_height is None:
            if hasattr(self, 'current_eye_roi') and self.current_eye_roi is not None:
                self.eye_height, self.eye_width = self.current_eye_roi.shape[:2]
                print(f"Göz ROI boyutları güncellendi: {self.eye_width}x{self.eye_height}")
            else:
                # Varsayılan değerler
                self.eye_width = 100
                self.eye_height = 50
                print(f"Varsayılan göz ROI boyutları kullanılıyor: {self.eye_width}x{self.eye_height}")
            
        # Yatay hareket (sağ-sol) için pupil pozisyonunu merkeze göre normalize et
        center_x = self.eye_width / 2
        
        # ÖNEMLİ DEĞİŞİKLİK: Göz sağa bakıyorsa mouse sağa, sola bakıyorsa mouse sola gitsin
        # Bu yüzden dx yönünü TERSİNE ÇEVİRMİYORUZ (eski kod: dx = -dx)
        dx = (pupil_pos[0] - center_x) / center_x  # Merkeze göre normalize et
        
        # Dikey hareket (yukarı-aşağı) için göz kapağı kapalılığını kullan
        if eye_aspect_ratio is not None:
            # Göz açıklık oranını kullan
            if not hasattr(self, 'base_eye_ratio') or self.base_eye_ratio is None:
                # İlk kez ölçülüyorsa, referans değeri kaydet
                self.base_eye_ratio = eye_aspect_ratio
                print(f"Temel göz oranı ayarlandı: {self.base_eye_ratio}")
                dy = 0
            else:
                # DÜZELTME: Göz daha açık (büyük oran) ise yukarı, daha kapalı (küçük oran) ise aşağı
                # Yönü düzeltiyoruz - oranları doğrudan kullan, tersine çevirme
                dy = (eye_aspect_ratio - self.base_eye_ratio) * 5.0
                print(f"Göz oranı değişimi: {eye_aspect_ratio} - {self.base_eye_ratio} = {(eye_aspect_ratio - self.base_eye_ratio)} (ölçeklemeden önce)")
        else:
            # Eğer göz açıklık oranı yoksa, y pozisyonunu kullan
            center_y = self.eye_height / 2
            # Burada da yönü tersine çevirmiyoruz
            dy = (pupil_pos[1] - center_y) / center_y
            print(f"Y hareket (konum bazlı): {dy}")
        
        # Değerleri sınırla ve ölçekle - hassasiyeti artır
        dx = max(-1.0, min(1.0, dx * 3.0))  # Hassasiyeti artırdık
        dy = max(-1.0, min(1.0, dy * 3.0))  # Hassasiyeti artırdık
        
        print(f"Hesaplanan hareket vektörü: dx={dx:.2f}, dy={dy:.2f}")
        return (dx, dy)

    def reset_pupil_position(self):
        self.initial_pupil_pos = None
