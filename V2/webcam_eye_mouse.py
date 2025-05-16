import cv2
import numpy as np
import dlib
import pyautogui
import math
import time
from collections import deque

# Ekran çözünürlüğünü al
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False  # Fare kenarlardan çıkarsa hata vermesini engeller

# Modelleri yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fare kontrol parametreleri
smoothing_factor = 0.7
smoothing_window = 5
move_speed = 2.0
click_hold_time = 0.5  # saniye olarak göz kırpma süre eşiği
blink_counter = 0
last_blink_time = time.time()
is_clicking = False

# Göz pozisyonlarını yumuşatma için
eye_position_history = deque(maxlen=smoothing_window)

# Göz çevresi noktaları
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

def get_eye_aspect_ratio(eye_points, facial_landmarks):
    # Göz köşelerindeki noktaları al
    corner_left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    
    # Üst ve alt göz kapağı noktaları
    top1 = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
    top2 = (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y)
    bottom1 = (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y)
    bottom2 = (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
    
    # Dikey mesafeleri hesapla
    d1 = math.dist(top1, bottom1)
    d2 = math.dist(top2, bottom2)
    
    # Yatay mesafeyi hesapla
    d3 = math.dist(corner_left, corner_right)
    
    # Göz aspect oranı (EAR)
    ear = (d1 + d2) / (2.0 * d3)
    
    return ear

def get_pupil_position(eye_points, facial_landmarks, frame, gray):
    # Göz bölgesini çıkart
    region_points = [(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points]
    min_x = min(pt[0] for pt in region_points)
    max_x = max(pt[0] for pt in region_points)
    min_y = min(pt[1] for pt in region_points)
    max_y = max(pt[1] for pt in region_points)
    
    # Göz bölgesine biraz kenar boşluğu ekle
    padding = 5
    eye_region = gray[max(0, min_y-padding):min(gray.shape[0], max_y+padding), 
                       max(0, min_x-padding):min(gray.shape[1], max_x+padding)]
    
    if eye_region.size == 0:
        return None, None
    
    # Göz bölgesini iyileştir
    eye_region = cv2.equalizeHist(eye_region)
    eye_region = cv2.GaussianBlur(eye_region, (7, 7), 0)
    
    # Göz bebeğini bul (en karanlık bölge)
    _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)  # 40 yerine 30 kullandık
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # En büyük kontur (muhtemelen göz bebeği)
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    
    if M["m00"] == 0:
        return None, None
    
    # Göz bebeğinin merkez noktasını hesapla
    cx = int(M["m10"] / M["m00"]) + min_x - padding
    cy = int(M["m01"] / M["m00"]) + min_y - padding
    
    # Görselleştirme için (hem tespit edilen göz bebeğini hem de eşik değeri görüntüsünü göster)
    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    
    # Hata ayıklama: Eşik değeri görüntüsünü göster
    if eye_region.size > 0:
        thresh_resized = cv2.resize(thresh, (100, 50))
        frame[10:60, frame.shape[1]-110:frame.shape[1]-10] = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
    
    # Göz merkezi
    eye_center_x = (min_x + max_x) // 2
    eye_center_y = (min_y + max_y) // 2
    
    # Göz bebeği konumunu normalize et (-1 ile 1 arasında)
    x_ratio = 2.5 * (cx - min_x) / (max_x - min_x) - 1.25  # Daha geniş aralık
    y_ratio = 2.5 * (cy - min_y) / (max_y - min_y) - 1.25  # Daha geniş aralık
    
    # Debug için konumu göster
    cv2.putText(frame, f"X: {x_ratio:.2f}, Y: {y_ratio:.2f}", 
                (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Göz çerçevesi içindeki y pozisyonunu görsel olarak göster
    mid_x = (min_x + max_x) // 2
    y_pos = min_y + int((y_ratio + 1.25) / 2.5 * (max_y - min_y))
    cv2.line(frame, (mid_x-10, y_pos), (mid_x+10, y_pos), (0, 0, 255), 2)
    
    return x_ratio, y_ratio

def move_mouse_smooth(target_x, target_y):
    current_x, current_y = pyautogui.position()
    new_x = current_x + (target_x - current_x) * smoothing_factor
    new_y = current_y + (target_y - current_y) * smoothing_factor
    pyautogui.moveTo(int(new_x), int(new_y))

def main():
    global blink_counter, last_blink_time, is_clicking
    
    cap = cv2.VideoCapture(0)
    
    # Kamera parametrelerini ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Görüntüyü çevir (ayna etkisi)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri algıla
        faces = detector(gray)
        
        if len(faces) > 0:
            # İlk yüz için
            face = faces[0]
            landmarks = predictor(gray, face)
            
            # Her iki göz için aspect oranı hesapla
            left_ear = get_eye_aspect_ratio(LEFT_EYE_POINTS, landmarks)
            right_ear = get_eye_aspect_ratio(RIGHT_EYE_POINTS, landmarks)
            
            # İki gözün ortalaması
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Göz kırpma tespiti
            EAR_THRESHOLD = 0.2
            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
                if blink_counter >= 3 and not is_clicking and time.time() - last_blink_time > 1.0:
                    # Uzun bir göz kırpma (tıklama)
                    is_clicking = True
                    pyautogui.mouseDown()
                    cv2.putText(frame, "CLICK!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if blink_counter > 0:
                    last_blink_time = time.time()
                blink_counter = 0
                if is_clicking:
                    is_clicking = False
                    pyautogui.mouseUp()
            
            # Her iki göz için göz bebeği konumunu tespit et
            left_x, left_y = get_pupil_position(LEFT_EYE_POINTS, landmarks, frame, gray)
            right_x, right_y = get_pupil_position(RIGHT_EYE_POINTS, landmarks, frame, gray)
            
            # Göz bebeği konumu ile göz açıklık oranını birleştirerek fare konumunu hesapla
            if left_x is not None and right_x is not None:
                # İki gözün ortalaması
                avg_x = (left_x + right_x) / 2
                avg_y = (left_y + right_y) / 2
                
                # İki göz arasındaki farkı kullanarak ekstra yatay hareket ekle
                eye_diff = left_x - right_x  # Sol göz - sağ göz farkı
                
                if abs(eye_diff) > 0.2:
                    avg_x = avg_x + eye_diff * 0.5
                
                # Y ekseni için özel işlem
                # Y koordinat sistemini TERS çevirelim - yukarı bakmak ekranın üstü olsun
                # -1 ile çarparak Y koordinat sistemini tersine çeviriyoruz
                avg_y = -avg_y
                
                # Debug bilgisi
                cv2.putText(frame, f"Raw Y: {avg_y:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                EAR_MIN = 0.01  # Minimum EAR değeri (göz tamamen kapalıyken)
                EAR_MAX = 0.28
                # Göz açıklık oranını Y pozisyonu için bir faktör olarak kullan
                ear_normal = (avg_ear - EAR_MIN) / (EAR_MAX - EAR_MIN) - 0.8
                ear_normal = max(0, min(1, ear_normal))
                
                # Göz açıklık değeri ve y pozisyonunu birleştir
                # NOT: ear_normal değerini Y değerini AŞAĞI kaydırmak için kullanıyoruz
                y_combined = (avg_y * 0.3) + ((1 - ear_normal) * 0.7)
                
                # Ekran koordinatlarına çevir - ölçeklendirme ve offset değerlerini değiştirdik
                target_x = SCREEN_WIDTH * (avg_x * 1.2 + 0.5)

                # Y ekseninde daha az ölçeklendirme ve daha düşük bir offset kullanıyoruz
                y_offset = 0.3  # Ekranın ortasına yakın bir nokta
                y_scale = 0.8  # Daha küçük bir ölçek - aşırı değerleri engellemek için

                target_y = SCREEN_HEIGHT * (y_combined * y_scale + y_offset) - 300

                # Debug: Y hedef değerini ve ekran yüksekliğini göster
                cv2.putText(frame, f"Target Y: {target_y:.0f}/{SCREEN_HEIGHT}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Değerleri ekran sınırları içinde tut
                target_x = max(0, min(SCREEN_WIDTH-1, target_x))
                target_y = max(0, min(SCREEN_HEIGHT-1, target_y))
                
                # Debug bilgilerini göster
                cv2.putText(frame, f"EAR Norm: {ear_normal:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Y Comb: {y_combined:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Fare konumunu yumuşat
                eye_position_history.append((target_x, target_y))
                if len(eye_position_history) == smoothing_window:
                    avg_x = sum(pos[0] for pos in eye_position_history) / smoothing_window
                    avg_y = sum(pos[1] for pos in eye_position_history) / smoothing_window
                    move_mouse_smooth(avg_x, avg_y)
            
            # Göz aspect oranını ekrana yazdır
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Göz çerçevelerini çiz
            for eye_points in [LEFT_EYE_POINTS, RIGHT_EYE_POINTS]:
                pts = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points], np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
        
        # Fare konumu bilgisini ekranda göster
        mouse_x, mouse_y = pyautogui.position()
        cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Görüntüyü göster
        cv2.imshow("Webcam Eye Mouse", frame)
        
        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()