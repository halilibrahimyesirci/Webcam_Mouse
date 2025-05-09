import cv2
import numpy as np
import pyautogui
from threading import Thread
import time
import keyboard
from src.utils.face_detector import FaceDetector
from src.utils.eye_tracker import EyeTracker
from src.gui.gui_manager import WebcamMouseControlGUI
from src.utils.calibration_helper import CalibrationHelper
import tkinter as tk
import customtkinter as ctk
import os
import dlib
import math

class MouseController:
    def __init__(self):
        # Initialize GUI
        self.root = ctk.CTk()
        self.gui = WebcamMouseControlGUI(self.root)
        # GUI'ye controller referansını gönder
        self.gui._controller = self
        
        # Initialize detectors
        models_dir = os.path.join('src', 'models')
        face_cascade_path = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        shape_predictor_path = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = False  # Disable fail-safe for this application
        
        print(f"Loading face detector (dlib by default)")
        self.face_detector = FaceDetector(cascade_path=face_cascade_path, use_dlib=True)
        print(f"Loading eye landmark predictor from: {os.path.abspath(shape_predictor_path)}")
        self.eye_tracker = EyeTracker(predictor_path=shape_predictor_path)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
          # Mouse control state
        self.screen_width, self.screen_height = pyautogui.size()
        self.selected_eye_type = None
        self.selected_pupil = None
        
        self.mouse_control_active = True  # Mouse kontrolü başlangıçta aktif
        
        # Calibration data
        self.calibration_mode = False
        self.calibration_points = [
            (50, 50),                                   # sol üst
            (self.screen_width - 50, 50),               # sağ üst
            (self.screen_width//2, self.screen_height//2), # orta
            (50, self.screen_height - 50),              # sol alt
            (self.screen_width - 50, self.screen_height - 50)  # sağ alt
        ]
        self.current_calibration_point = 0
        self.calibration_data = {
            'points': [],  # Her noktada göz açıklık ve konum verisi
            'completed': False
        }
        
        self.calibration_helper = CalibrationHelper()
        
        # Setup mouse click handling
        self.setup_keyboard_control()
        
        # Start processing thread
        self.start_processing()
        
    def setup_keyboard_control(self):
        keyboard.on_press_key('esc', self.check_exit_combination)
        keyboard.on_press_key('l', self.check_exit_combination)
        self.last_key_time = 0
        self.keys_pressed = set()
        
    def toggle_mouse_control(self):
        self.mouse_control_active = not self.mouse_control_active
        status = "Aktif" if self.mouse_control_active else "Durduruldu"
        self.gui.status_label.configure(text=f"Mouse Kontrolü: {status}")
        
        # GUI butonunu güncelle eğer varsa
        if hasattr(self.gui, 'mouse_control_btn'):
            if self.mouse_control_active:
                self.gui.mouse_control_btn.configure(
                    text="Mouse Kontrolünü Durdur",
                    fg_color="#c7503e",
                    hover_color="#a73e30"
                )
            else:
                self.gui.mouse_control_btn.configure(
                    text="Mouse Kontrolünü Başlat",
                    fg_color="#3a7ebf", 
                    hover_color="#306998"
                )
        
        print(f"Mouse kontrolü {status}")
        
    def check_exit_combination(self, event):
        current_time = time.time()
        self.keys_pressed.add(event.name)
        
        if current_time - self.last_key_time > 0.5:
            self.keys_pressed.clear()
            
        self.last_key_time = current_time
        print(f"Basılan tuşlar: {self.keys_pressed}")
        
        if 'esc' in self.keys_pressed and 'l' in self.keys_pressed:
            print("ESC+L basıldı, mouse kontrolü değiştiriliyor...")
            # Sadece mouse kontrolünü durdur, programı kapatma
            self.toggle_mouse_control()
        
    def process_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # Check for user clicks on the video frames
            main_click, eye_click = self.gui.get_mouse_click_coords()
            
            # Check if calibration mode was toggled
            if hasattr(self.gui, 'calibration_active') and self.gui.calibration_active != self.calibration_mode:
                self.calibration_mode = self.gui.calibration_active
                if self.calibration_mode:
                    self.current_calibration_point = 0
                    # Reset calibration data
                    self.calibration_data = {
                        'points': [],
                        'completed': False
                    }
                    # Move mouse to first calibration point
                    pyautogui.moveTo(self.calibration_points[0][0], self.calibration_points[0][1])
            
            # Handle calibration if active
            if self.calibration_mode and not self.calibration_data['completed']:
                # Process calibration
                self.calibration_helper.process_calibration(frame, self)
            else:
                # Normal tracking mode
                # Detect face
                face_coords_dlib_or_haar = self.face_detector.detect_face(frame)
                self.face_detector.draw_face(frame, face_coords_dlib_or_haar)
                
                if face_coords_dlib_or_haar is not None:
                    face_rect_dlib = None
                    if isinstance(face_coords_dlib_or_haar, dlib.rectangle):
                        face_rect_dlib = face_coords_dlib_or_haar
                    elif self.face_detector.use_dlib is False and face_coords_dlib_or_haar is not None:
                        x, y, w, h = face_coords_dlib_or_haar
                        face_rect_dlib = dlib.rectangle(x, y, x + w, y + h)

                    if face_rect_dlib:
                        face_roi_for_display = self.face_detector.get_face_roi(frame, face_rect_dlib)

                        if self.selected_eye_type is None:
                            left_eye_pts, right_eye_pts = self.eye_tracker.get_eye_landmarks(frame, face_rect_dlib)
                            
                            # Draw landmarks for both eyes for selection
                            left_eye_center = None
                            right_eye_center = None
                            
                            if left_eye_pts is not None:
                                for pt in left_eye_pts:
                                    cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
                                # Calculate center of left eye for selection
                                left_eye_center = (np.mean(left_eye_pts[:, 0]).astype(int), 
                                                  np.mean(left_eye_pts[:, 1]).astype(int))
                                # Draw a highlighted circle at eye center for better visibility
                                cv2.circle(frame, left_eye_center, 5, (0, 255, 0), 2)
                                cv2.putText(frame, "L", (left_eye_center[0]-10, left_eye_center[1]-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            if right_eye_pts is not None:
                                for pt in right_eye_pts:
                                    cv2.circle(frame, tuple(pt), 2, (0, 0, 255), -1)
                                # Calculate center of right eye for selection
                                right_eye_center = (np.mean(right_eye_pts[:, 0]).astype(int), 
                                                   np.mean(right_eye_pts[:, 1]).astype(int))
                                # Draw a highlighted circle at eye center for better visibility
                                cv2.circle(frame, right_eye_center, 5, (0, 0, 255), 2)
                                cv2.putText(frame, "R", (right_eye_center[0]-10, right_eye_center[1]-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                            # Handle user click for eye selection
                            if main_click is not None and self.gui.is_eye_selection_mode:
                                click_x, click_y = main_click
                                # Convert GUI coordinates to frame coordinates (this may need adjustment)
                                # For simplicity, assume click and frame coordinates match for now
                                
                                # Check if click is near left eye
                                if left_eye_center and right_eye_center:
                                    # Calculate distances to both eye centers
                                    left_dist = np.sqrt((click_x - left_eye_center[0])**2 + (click_y - left_eye_center[1])**2)
                                    right_dist = np.sqrt((click_x - right_eye_center[0])**2 + (click_y - right_eye_center[1])**2)
                                    
                                    # Select the closest eye
                                    if left_dist < right_dist:
                                        self.selected_eye_type = 'left'
                                        self.eye_tracker.select_eye(frame, left_eye_pts, is_left_eye=True)
                                        self.gui.status_label.configure(text="Left eye selected. Select pupil.")
                                    else:
                                        self.selected_eye_type = 'right'
                                        self.eye_tracker.select_eye(frame, right_eye_pts, is_left_eye=False)
                                        self.gui.status_label.configure(text="Right eye selected. Select pupil.")
                                    
                                    self.gui.enable_pupil_selection()
                                    self.gui.is_eye_selection_mode = False
                                    self.eye_tracker.reset_pupil_position()
                        else:
                            # Eye already selected, track pupil
                            eye_landmarks_to_track = None
                            is_left = self.selected_eye_type == 'left'
                            
                            current_left_eye_pts, current_right_eye_pts = self.eye_tracker.get_eye_landmarks(frame, face_rect_dlib)
                            
                            if is_left and current_left_eye_pts is not None:
                                eye_landmarks_to_track = current_left_eye_pts
                            elif not is_left and current_right_eye_pts is not None:
                                eye_landmarks_to_track = current_right_eye_pts
                            
                            if eye_landmarks_to_track is not None:
                                eye_frame_roi = self.eye_tracker.select_eye(frame, eye_landmarks_to_track, is_left_eye=is_left)
                                if eye_frame_roi is not None:
                                    self.gui.update_eye_zoom(eye_frame_roi)
                                    
                                    # Handle pupil selection by user click
                                    if eye_click is not None and self.gui.is_pupil_selection_mode:
                                        # Convert GUI coordinates to eye ROI coordinates
                                        pupil_x, pupil_y = eye_click
                                        # Set initial pupil position for tracking
                                        self.eye_tracker.initial_pupil_pos = (pupil_x, pupil_y)
                                        self.gui.status_label.configure(text=f"Pupil selected. Tracking enabled.")
                                        self.gui.is_pupil_selection_mode = False
                                    
                                    # Process pupil detection and movement
                                    if not self.gui.is_pupil_selection_mode or self.eye_tracker.initial_pupil_pos is not None:
                                        pupil_pos_in_roi = self.eye_tracker.detect_pupil(eye_frame_roi)
                                        
                                        # Göz açıklık bilgisini hesapla
                                        eye_height = np.max(eye_landmarks_to_track[:, 1]) - np.min(eye_landmarks_to_track[:, 1])
                                        eye_width = np.max(eye_landmarks_to_track[:, 0]) - np.min(eye_landmarks_to_track[:, 0])
                                        eye_aspect_ratio = eye_height / eye_width if eye_width > 0 else 0
                                        
                                        # Kullanıcı pupil seçmese bile ilk algılanan pupil pozisyonunu başlangıç noktası olarak kullan
                                        if self.eye_tracker.initial_pupil_pos is None and pupil_pos_in_roi is not None:
                                            self.eye_tracker.initial_pupil_pos = pupil_pos_in_roi
                                            print(f"İlk pupil konumu otomatik ayarlandı: {pupil_pos_in_roi}")
                                            self.gui.status_label.configure(text=f"Pupil otomatik algılandı. Tracking enabled.")
                                            self.gui.is_pupil_selection_mode = False
                                        
                                        if pupil_pos_in_roi is not None:
                                            movement = self.eye_tracker.calculate_pupil_movement(
                                                pupil_pos_in_roi, 
                                                eye_height=eye_height,
                                                eye_aspect_ratio=eye_aspect_ratio
                                            )
                                            
                                            
                                            dx = movement[0]  # Yönü tersine çevir

                                            # DÜZELTME: Yukarı/aşağı yönünü düzelt
                                            dy = -movement[1]  # Yönü tersine çevir

                                            # Debug bilgisi ekle
                                            print(f"Hesaplanan hareket: dx={dx:.2f}, dy={dy:.2f}")
                                            
                                            # Sadece mouse kontrolü aktifse hareket ettir
                                            if self.mouse_control_active:
                                                print("Mouse kontrolü aktif, hareket ettiriliyor...")
                                                self.move_mouse((dx, dy))
                                            elif not self.mouse_control_active:
                                                print("Mouse kontrolü devre dışı, hareket ettirilmiyor.")
            
            self.gui.update_main_video(frame)
            
    def move_mouse(self, movement):
        if movement is None:
            print("Hareket parametresi None, hareket ettirilemiyor.")
            return
            
        # Debug bilgisi ekle
        print(f"Mouse hareketi: {movement}, Kontrol aktif: {self.mouse_control_active}")
        
        if not self.mouse_control_active:
            print("Mouse kontrolü devre dışı olduğu için hareket ettirilmiyor.")
            return
            
        dx_normalized, dy_normalized = movement
        
        # Hassasiyet değerlerini al
        try:
            sensitivities = self.gui.get_sensitivities()
        except Exception as e:
            print(f"Hassasiyet değerlerini alırken hata: {e}")
            sensitivities = {'x': 1.0, 'y': 1.0, 'smoothing': 0.3, 'eyelid': 1.0}
        
        # Ekran ölçeği ve hassasiyeti ayarla - hassasiyeti artır
        screen_movement_scale_x = self.screen_width * 0.1  # 0.05'ten 0.1'e artırdık
        screen_movement_scale_y = self.screen_height * 0.1  # 0.05'ten 0.1'e artırdık
        
        # Hassasiyet değerleri uygula
        scaled_dx = dx_normalized * screen_movement_scale_x * sensitivities['x']
        scaled_dy = dy_normalized * screen_movement_scale_y * sensitivities['y']
        
        # Debug bilgisi
        print(f"Hassasiyet değerleri: {sensitivities}")
        print(f"Ham hareket: dx={dx_normalized:.2f}, dy={dy_normalized:.2f}")
        print(f"Ölçeklenmiş hareket: dx={scaled_dx:.2f}, dy={scaled_dy:.2f}")
        
        # Mevcut mouse konumu
        current_x, current_y = pyautogui.position()
        
        # Hedef konum
        target_x = current_x + scaled_dx
        target_y = current_y + scaled_dy
        
        # Yumuşatma (smoothing) uygula - daha hızlı tepki için
        smoothing = min(sensitivities.get('smoothing', 0.5), 1.0)  # Varsayılan değeri 0.3'ten 0.5'e artırdık
        new_x = current_x + (target_x - current_x) * smoothing
        new_y = current_y + (target_y - current_y) * smoothing
        
        # Debug
        print(f"Mevcut konum: ({current_x}, {current_y})")
        print(f"Hedef konum: ({target_x}, {target_y})")
        print(f"Yeni konum: ({new_x}, {new_y})")
        
        # Ekran sınırlarını kontrol et
        new_x = max(0, min(new_x, self.screen_width - 1))
        new_y = max(0, min(new_y, self.screen_height - 1))
        
        # Mouse'u hareket ettir
        try:
            pyautogui.moveTo(new_x, new_y)
            print(f"Mouse başarıyla şu konuma taşındı: ({new_x}, {new_y})")
        except Exception as e:
            print(f"Mouse hareketi sırasında hata: {e}")
            
    def start_processing(self):
        self.processing_thread = Thread(target=self.process_frame, daemon=True)
        self.processing_thread.start()
        
    def stop_application(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.quit()
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.stop_application)
        self.root.mainloop()

def main():
    try:
        controller = MouseController()
        controller.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
