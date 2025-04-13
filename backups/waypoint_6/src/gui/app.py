import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
from pynput import keyboard
from tracking.eye_tracker import EyeTracker
from calibration.calibrator import Calibrator
from utils.mouse_controller import MouseController
from utils.profile_manager import ProfileManager
from gui.calibration_window import CalibrationWindow

class EyeTrackingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Eye Tracking Mouse Control")
        self.geometry("1024x768")
        
        # Ekran boyutlarını al
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        
        # Initialize components
        self.eye_tracker = EyeTracker()
        self.calibrator = Calibrator()
        self.mouse_controller = MouseController()
        self.profile_manager = ProfileManager()  # Profil yöneticisini ekle
        self.is_tracking = False
        self.is_calibrating = False
        self.mouse_control_enabled = False
        
        # Keyboard shortcut state
        self.esc_pressed = False
        
        # Initialize keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.keyboard_listener.start()
        
        self._setup_ui()
        self._load_profiles()  # Profilleri yükle
        
        # Bind closing event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_ui(self):
        # Create main container with grid 2x2
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=1)
        
        # Camera frame
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Controls frame
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Camera controls
        self.camera_select = ctk.CTkComboBox(
            self.controls_frame,
            values=self._get_available_cameras(),
            command=self._on_camera_select
        )
        self.camera_select.pack(padx=10, pady=10)
        
        self.start_button = ctk.CTkButton(
            self.controls_frame,
            text="Start Tracking",
            command=self._toggle_tracking
        )
        self.start_button.pack(padx=10, pady=10)
        
        # Calibration controls
        self.calibrate_button = ctk.CTkButton(
            self.controls_frame,
            text="Start Calibration",
            command=self._start_calibration
        )
        self.calibrate_button.pack(padx=10, pady=10)
        
        # Mouse control toggle
        self.mouse_control_button = ctk.CTkButton(
            self.controls_frame,
            text="Enable Mouse Control",
            command=self._toggle_mouse_control,
            state="disabled"
        )
        self.mouse_control_button.pack(padx=10, pady=10)
        
        # Smoothing control
        self.smoothing_frame = ctk.CTkFrame(self.controls_frame)
        self.smoothing_frame.pack(padx=10, pady=(10, 0), fill="x")
        
        self.smoothing_label = ctk.CTkLabel(
            self.smoothing_frame,
            text="Mouse Smoothing:"
        )
        self.smoothing_label.pack(side="left", padx=5)
        
        self.smoothing_value_label = ctk.CTkLabel(
            self.smoothing_frame,
            text="30%",
            width=40
        )
        self.smoothing_value_label.pack(side="right", padx=5)
        
        self.smoothing_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=self._update_smoothing
        )
        self.smoothing_slider.set(0.3)
        self.smoothing_slider.pack(padx=10, pady=(0, 10))
        
        # Momentum control
        self.momentum_frame = ctk.CTkFrame(self.controls_frame)
        self.momentum_frame.pack(padx=10, pady=(10, 0), fill="x")
        
        self.momentum_label = ctk.CTkLabel(
            self.momentum_frame,
            text="Momentum:"
        )
        self.momentum_label.pack(side="left", padx=5)
        
        self.momentum_value_label = ctk.CTkLabel(
            self.momentum_frame,
            text="50%",
            width=40
        )
        self.momentum_value_label.pack(side="right", padx=5)
        
        self.momentum_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=self._update_momentum
        )
        self.momentum_slider.set(0.5)
        self.momentum_slider.pack(padx=10, pady=(0, 10))
        
        # Vertical sensitivity control
        self.vertical_sens_frame = ctk.CTkFrame(self.controls_frame)
        self.vertical_sens_frame.pack(padx=10, pady=(10, 0), fill="x")
        
        self.vertical_sens_label = ctk.CTkLabel(
            self.vertical_sens_frame,
            text="Dikey Hassasiyet:"
        )
        self.vertical_sens_label.pack(side="left", padx=5)
        
        self.vertical_sens_value_label = ctk.CTkLabel(
            self.vertical_sens_frame,
            text="100%",
            width=40
        )
        self.vertical_sens_value_label.pack(side="right", padx=5)
        
        self.vertical_sens_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0.1,
            to=2.0,
            number_of_steps=100,
            command=self._update_vertical_sensitivity
        )
        self.vertical_sens_slider.set(1.0)
        self.vertical_sens_slider.pack(padx=10, pady=(0, 10))

        # Profile Management Frame
        self.profile_frame = ctk.CTkFrame(self.controls_frame)
        self.profile_frame.pack(padx=10, pady=10, fill="x")
        
        self.profile_label = ctk.CTkLabel(
            self.profile_frame,
            text="Kalibrasyon Profili:"
        )
        self.profile_label.pack(side="left", padx=5)
        
        self.profile_var = ctk.StringVar(value="Yeni Profil")
        self.profile_combo = ctk.CTkComboBox(
            self.profile_frame,
            values=["Yeni Profil"],
            variable=self.profile_var,
            command=self._on_profile_select
        )
        self.profile_combo.pack(side="left", padx=5)
        
        self.save_profile_button = ctk.CTkButton(
            self.profile_frame,
            text="Kaydet",
            command=self._save_current_profile,
            width=60
        )
        self.save_profile_button.pack(side="left", padx=5)
        
        self.delete_profile_button = ctk.CTkButton(
            self.profile_frame,
            text="Sil",
            command=self._delete_current_profile,
            width=60
        )
        self.delete_profile_button.pack(side="left", padx=5)

        # Status frame
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Debug canvas for gaze visualization
        self.debug_canvas = ctk.CTkCanvas(
            self.status_frame,
            width=400,
            height=100,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.debug_canvas.pack(side="left", padx=10, pady=10)
        
        # Draw border rectangle
        self.debug_canvas.create_rectangle(
            2, 2, 398, 98,
            outline="gray",
            width=1
        )
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready to start",
            font=("Helvetica", 14)
        )
        self.status_label.pack(side="right", padx=10, pady=10)
        
    def _update_smoothing(self, value):
        """Update mouse movement smoothing factor"""
        value = float(value)
        self.mouse_controller.set_smoothing(value)
        self.smoothing_value_label.configure(text=f"{int(value * 100)}%")
        
    def _update_momentum(self, value):
        """Update mouse movement momentum factor"""
        value = float(value)
        self.mouse_controller.set_momentum(value)
        self.momentum_value_label.configure(text=f"{int(value * 100)}%")
        
    def _update_vertical_sensitivity(self, value):
        """Update vertical sensitivity factor"""
        value = float(value)
        percentage = int((value - 0.1) / (2.0 - 0.1) * 100)  # 0.1 ile 2.0 arasını 0-100'e dönüştür
        self.mouse_controller.set_vertical_sensitivity(value)
        self.vertical_sens_value_label.configure(text=f"{percentage}%")
        
    def _toggle_mouse_control(self):
        """Toggle mouse control on/off"""
        if not self.mouse_control_enabled:
            if self.calibrator.transformation_matrices[1] is not None:  # Orta mesafe için kontrol
                self.mouse_control_enabled = True
                self.mouse_controller.start_control()
                self.mouse_control_button.configure(text="Disable Mouse Control")
                self.status_label.configure(text="Mouse control active")
            else:
                self.status_label.configure(text="Lütfen önce kalibrasyonu tamamlayın")
        else:
            self.mouse_control_enabled = False
            self.mouse_controller.stop_control()
            self.mouse_control_button.configure(text="Enable Mouse Control")
            self.status_label.configure(text="Mouse control disabled")
            
    def _update_frame(self):
        """Update camera frame with tracking visualization"""
        while self.is_tracking:
            ret, frame = self.eye_tracker.camera.read()
            if ret:
                tracking_result = self.eye_tracker.get_eye_position(frame)
                
                if tracking_result:
                    frame = tracking_result['original_frame']
                    gaze_point = tracking_result['gaze_point']
                    left_eye = tracking_result['left_eye']
                    right_eye = tracking_result['right_eye']
                    left_metrics = tracking_result['left_metrics']
                    right_metrics = tracking_result['right_metrics']
                    
                    # Koordinatları piksel koordinatlarına dönüştür
                    left_x = int(left_eye[0] * frame.shape[1])
                    left_y = int(left_eye[1] * frame.shape[0])
                    right_x = int(right_eye[0] * frame.shape[1])
                    right_y = int(right_eye[1] * frame.shape[0])
                    
                    # Her göz için elips çiz
                    self._draw_iris_visualization(frame, (left_x, left_y), left_metrics, (255, 0, 0))  # Mavi
                    self._draw_iris_visualization(frame, (right_x, right_y), right_metrics, (0, 0, 255))  # Kırmızı
                    
                    # Merkez noktasını çiz
                    center_x = (left_x + right_x) // 2
                    center_y = (left_y + right_y) // 2
                    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)  # Yeşil
                    
                    # Fare pozisyonunu güncelle
                    if self.mouse_control_enabled and not self.is_calibrating:
                        mapped_pos = self.calibrator.get_screen_position(gaze_point)
                        if mapped_pos is not None:
                            screen_x = mapped_pos[0] / self.calibrator.screen_width
                            screen_y = mapped_pos[1] / self.calibrator.screen_height
                            self.mouse_controller.move_mouse(screen_x, screen_y)
                            self._update_debug_visualization(mapped_pos)
                
                # Frame'i göster
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo

    def update(self):
        """Update eye tracking and mouse position"""
        if not self.eye_tracker.camera:
            return
            
        ret, frame = self.eye_tracker.camera.read()
        if not ret:
            return

        tracking_result = self.eye_tracker.get_eye_position(frame)
        if tracking_result is None:
            return

        # Debug görselleştirmesi
        debug_frame = tracking_result['debug_frame']
        debug_frame = cv2.resize(debug_frame, (640, 480))
        
        # OpenCV görüntüsünü CTk için hazırla
        image = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        # Görüntüyü güncelle
        if not hasattr(self, 'video_label'):
            self.video_label = ctk.CTkLabel(self, image=photo, text="")
            self.video_label.grid(row=0, column=1, padx=10, pady=10)
        else:
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        # Göz pozisyonunu güncelle
        gaze_point = tracking_result['gaze_point']
        depth = tracking_result['depth']

        # Profil seçiliyse mouse'u güncelle
        if self.current_profile and self.mouse_controller:
            self.mouse_controller.move_mouse(gaze_point, depth)
            
        self.after(10, self.update)  # 10ms sonra tekrar çağır

    def _draw_iris_visualization(self, frame, center, metrics, color):
        """İris elips ve metriklerini görselleştir"""
        # Elips boyutlarını piksel cinsinden hesapla
        size = int(metrics['normalized_size'] * 50)  # Göreceli boyut
        angle = metrics['angle']
        regularity = metrics['regularity']
        
        # Elips çiz
        axes = (size, int(size * regularity))  # Elips eksenleri
        cv2.ellipse(frame, center, axes, angle, 0, 360, color, 2)
        
        # Güvenilirlik göstergesi
        confidence_color = (0, int(255 * regularity), 0)  # Yeşil ton (güvenilirliğe göre)
        
    def _update_debug_visualization(self, screen_pos):
        """Update debug canvas with current gaze position"""
        # Clear previous point
        self.debug_canvas.delete("gaze_point")
        
        # Convert screen coordinates to canvas coordinates
        canvas_width = 396  # 400 - 2*2 for border
        canvas_height = 96  # 100 - 2*2 for border
        
        x = 2 + (screen_pos[0] / self.winfo_screenwidth()) * canvas_width
        y = 2 + (screen_pos[1] / self.winfo_screenheight()) * canvas_height
        
        # Draw new point
        self.debug_canvas.create_oval(
            x-4, y-4, x+4, y+4,
            fill="yellow",
            outline="white",
            tags="gaze_point"
        )
        
    def _start_calibration(self):
        """Start calibration process"""
        if not self.is_tracking:
            self.status_label.configure(text="Please start tracking first")
            return
            
        self.calibrator.reset()
        self.is_calibrating = True
        self.calibrate_button.configure(state="disabled")
        self.mouse_control_button.configure(state="disabled")
        
        # Create and show calibration window
        self.calibration_window = CalibrationWindow(
            self,
            self.calibrator,
            self.eye_tracker,
            self._on_calibration_complete
        )
        
    def _on_calibration_complete(self):
        """Handle calibration completion"""
        self.is_calibrating = False
        self.calibrate_button.configure(state="normal")
        self.mouse_control_button.configure(state="normal")
        self.status_label.configure(text="Calibration completed")
        # Yeni kalibrasyondan sonra "Yeni Profil"e geç
        self.profile_var.set("Yeni Profil")
        
    def _get_available_cameras(self):
        """Get list of available camera devices"""
        camera_list = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(str(i))
                cap.release()
        return camera_list if camera_list else ["0"]
        
    def _on_camera_select(self, choice):
        """Handle camera selection change"""
        if self.is_tracking:
            self._toggle_tracking()  # Stop current tracking
        self.eye_tracker.camera_index = int(choice)
        
    def _toggle_tracking(self):
        """Toggle eye tracking on/off"""
        if not self.is_tracking:
            if self.eye_tracker.start_camera():
                self.is_tracking = True
                self.start_button.configure(text="Stop Tracking")
                self.tracking_thread = threading.Thread(target=self._update_frame)
                self.tracking_thread.daemon = True
                self.tracking_thread.start()
                self.status_label.configure(text="Tracking active")
        else:
            self.is_tracking = False
            self.eye_tracker.stop_camera()
            self.start_button.configure(text="Start Tracking")
            self.status_label.configure(text="Tracking stopped")
            
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            if key == keyboard.Key.esc:
                self.esc_pressed = True
            elif self.esc_pressed and key.char.lower() == 'l':
                # ESC + L combination detected
                if self.mouse_control_enabled:
                    self._toggle_mouse_control()
        except AttributeError:
            pass

    def _on_key_release(self, key):
        """Handle key release events"""
        if key == keyboard.Key.esc:
            self.esc_pressed = False

    def on_closing(self):
        """Clean up resources before closing"""
        self.is_tracking = False
        if hasattr(self, 'eye_tracker'):
            self.eye_tracker.stop_camera()
        self.keyboard_listener.stop()
        self.destroy()
        
    def _load_profiles(self):
        """Mevcut profilleri yükle"""
        profiles = self.profile_manager.get_profiles()
        values = ["Yeni Profil"] + profiles
        self.profile_combo.configure(values=values)
        
    def _on_profile_select(self, choice):
        """Profil seçildiğinde çalışır"""
        if choice == "Yeni Profil":
            self.calibrator.reset()
            self.mouse_control_button.configure(state="disabled")
            return
            
        # Seçilen profili yükle
        calibration_data, transformation_matrices = self.profile_manager.load_profile(choice)
        if calibration_data and transformation_matrices:
            self.calibrator.calibration_data = calibration_data
            self.calibrator.transformation_matrices = transformation_matrices
            self.mouse_control_button.configure(state="normal")
            self.status_label.configure(text=f"Profil yüklendi: {choice}")
        
    def _save_current_profile(self):
        """Mevcut kalibrasyonu profil olarak kaydet"""
        # Orta mesafe için kalibrasyon kontrolü
        if 1 not in self.calibrator.transformation_matrices or self.calibrator.transformation_matrices[1] is None:
            self.status_label.configure(text="Önce kalibrasyon yapmalısınız")
            return
            
        # Yeni profil adını al
        dialog = ctk.CTkInputDialog(
            text="Profil adını girin:",
            title="Profili Kaydet"
        )
        profile_name = dialog.get_input()
        
        if profile_name:
            self.profile_manager.save_profile(
                profile_name,
                self.calibrator.calibration_data,
                self.calibrator.transformation_matrices
            )
            self._load_profiles()
            self.profile_var.set(profile_name)
            self.status_label.configure(text=f"Profil kaydedildi: {profile_name}")
        
    def _delete_current_profile(self):
        """Seçili profili sil"""
        current_profile = self.profile_var.get()
        if current_profile == "Yeni Profil":
            return
            
        self.profile_manager.delete_profile(current_profile)
        self._load_profiles()
        self.profile_var.set("Yeni Profil")
        self.status_label.configure(text=f"Profil silindi: {current_profile}")