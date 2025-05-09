import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class WebcamMouseControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Mouse Control")
        self.root.geometry("1200x800")
        
        # Create frames
        self.setup_frames()
        
        # Control variables
        self.is_eye_selection_mode = False
        self.is_pupil_selection_mode = False
        
        # Mouse click handlers for video frames
        self.main_video_click_coords = None
        self.eye_video_click_coords = None
        
    def setup_frames(self):
        # Main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left control panel
        self.control_frame = ctk.CTkFrame(self.main_container)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right video panel
        self.video_container = ctk.CTkFrame(self.main_container)
        self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup control panel
        self.setup_control_panel()
        
        # Setup video displays
        self.setup_video_displays()
        
    def setup_control_panel(self):
        # Title
        ctk.CTkLabel(self.control_frame, text="Settings", font=("Arial", 20)).pack(pady=10)
        
        # Sensitivity controls
        self.create_sensitivity_controls(self.control_frame)
        
        # Select Eye Button
        self.select_eye_btn = ctk.CTkButton(
            self.control_frame,
            text="Select Eye",
            command=self.toggle_eye_selection
        )
        self.select_eye_btn.pack(pady=10)
        
        # Select Pupil Button
        self.select_pupil_btn = ctk.CTkButton(
            self.control_frame,
            text="Select Pupil",
            command=self.toggle_pupil_selection,
            state="disabled"
        )
        self.select_pupil_btn.pack(pady=10)
        
        # Calibration Button
        self.calibration_btn = ctk.CTkButton(
            self.control_frame,
            text="Start Calibration",
            command=self.toggle_calibration
        )
        self.calibration_btn.pack(pady=10)
        
        # Status Label
        self.status_label = ctk.CTkLabel(self.control_frame, text="Status: Waiting for eye selection")
        self.status_label.pack(pady=10)
        
        # Control Buttons
        self.create_control_buttons(self.control_frame)
        
    def create_sensitivity_controls(self, frame):
        # Yatay hassasiyet (sağ-sol)
        ttk.Label(frame, text="Yatay Hassasiyet (Sağ-Sol):").pack(anchor="w", padx=5)
        self.x_sensitivity = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, 
                                     orient="horizontal", length=200)
        self.x_sensitivity.set(1.0)
        self.x_sensitivity.pack(fill="x", padx=5, pady=5)
        
        # Dikey hassasiyet (yukarı-aşağı - göz açıklığı için)
        ttk.Label(frame, text="Dikey Hassasiyet (Yukarı-Aşağı):").pack(anchor="w", padx=5)
        self.y_sensitivity = tk.Scale(frame, from_=0.1, to=2.0, resolution=0.1, 
                                     orient="horizontal", length=200)
        self.y_sensitivity.set(1.0)
        self.y_sensitivity.pack(fill="x", padx=5, pady=5)
        
        # Yumuşatma (hareket gecikmesi)
        ttk.Label(frame, text="Yumuşatma:").pack(anchor="w", padx=5)
        self.smoothing = tk.Scale(frame, from_=0.1, to=1.0, resolution=0.1, 
                                 orient="horizontal", length=200)
        self.smoothing.set(0.3)
        self.smoothing.pack(fill="x", padx=5, pady=5)
        
        # Göz kapağı hassasiyeti (yukarı-aşağı için)
        ttk.Label(frame, text="Göz Kapağı Hassasiyeti:").pack(anchor="w", padx=5)
        self.eyelid_sensitivity = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1, 
                                        orient="horizontal", length=200)
        self.eyelid_sensitivity.set(1.0)
        self.eyelid_sensitivity.pack(fill="x", padx=5, pady=5)
        
    def create_control_buttons(self, parent_frame):
        buttons_frame = ttk.Frame(parent_frame)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        # Mouse kontrolü başlatma/durdurma butonu
        self.mouse_control_btn = ctk.CTkButton(
            buttons_frame, 
            text="Mouse Kontrolünü Durdur", 
            command=self.toggle_mouse_control,
            fg_color="#c7503e",
            hover_color="#a73e30",
            width=180
        )
        self.mouse_control_btn.pack(pady=5)
        
    def setup_video_displays(self):
        # Main video feed with click interaction
        self.main_video_frame = ctk.CTkFrame(self.video_container)
        self.main_video_frame.pack(expand=True, fill=tk.BOTH)
        
        self.main_video_label = ctk.CTkLabel(self.main_video_frame, text="")
        self.main_video_label.pack(expand=True, fill=tk.BOTH)
        # Bind click events to the main video label
        self.main_video_label.bind("<Button-1>", self.on_main_video_click)
        
        # Eye zoom frame with click interaction
        self.eye_zoom_frame = ctk.CTkFrame(self.video_container)
        self.eye_zoom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        self.eye_zoom_label = ctk.CTkLabel(self.eye_zoom_frame, text="")
        self.eye_zoom_label.pack(expand=True, fill=tk.BOTH)
        # Bind click events to the eye zoom label
        self.eye_zoom_label.bind("<Button-1>", self.on_eye_zoom_click)
        
    def on_main_video_click(self, event):
        """Handle clicks on the main video (for eye selection)"""
        if self.is_eye_selection_mode:
            self.main_video_click_coords = (event.x, event.y)
            self.status_label.configure(text=f"Click position: {event.x}, {event.y}")
            print(f"Main video clicked at: {event.x}, {event.y}")
            
    def on_eye_zoom_click(self, event):
        """Handle clicks on the eye zoom view (for pupil selection)"""
        if self.is_pupil_selection_mode:
            self.eye_video_click_coords = (event.x, event.y)
            self.status_label.configure(text=f"Pupil selected at: {event.x}, {event.y}")
            print(f"Eye zoom video clicked at: {event.x}, {event.y}")
            # Pupil selection is complete
            self.is_pupil_selection_mode = False
            self.select_pupil_btn.configure(text="Select Pupil")
        
    def toggle_eye_selection(self):
        self.is_eye_selection_mode = not self.is_eye_selection_mode
        if self.is_eye_selection_mode:
            self.select_eye_btn.configure(text="Cancel Eye Selection")
            self.status_label.configure(text="Click on an eye in the video")
        else:
            self.select_eye_btn.configure(text="Select Eye")
            self.status_label.configure(text="Eye selection cancelled")
            
    def toggle_pupil_selection(self):
        self.is_pupil_selection_mode = not self.is_pupil_selection_mode
        if self.is_pupil_selection_mode:
            self.select_pupil_btn.configure(text="Cancel Pupil Selection")
            self.status_label.configure(text="Click on the pupil in the zoomed eye view")
        else:
            self.select_pupil_btn.configure(text="Select Pupil")
            self.status_label.configure(text="Pupil selection cancelled")
            
    def enable_pupil_selection(self):
        self.select_pupil_btn.configure(state="normal")
        
    def update_main_video(self, frame):
        if frame is not None:
            frame = cv2.resize(frame, (800, 600))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Convert to CTkImage to avoid HighDPI warning
            img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
            self.main_video_label.configure(image=img_ctk)
            self.main_video_label.image = img_ctk
            
    def update_eye_zoom(self, frame):
        if frame is not None:
            # Resize eye frame to be larger
            frame = cv2.resize(frame, (200, 150))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Convert to CTkImage to avoid HighDPI warning
            img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 150))
            self.eye_zoom_label.configure(image=img_ctk)
            self.eye_zoom_label.image = img_ctk
            
    def get_sensitivities(self):
        return {
            'x': self.x_sensitivity.get(),
            'y': self.y_sensitivity.get(), 
            'smoothing': self.smoothing.get(),
            'eyelid': self.eyelid_sensitivity.get() if hasattr(self, 'eyelid_sensitivity') else 1.0
        }
        
    def get_mouse_click_coords(self):
        """Return click coordinates and reset them"""
        main_coords = self.main_video_click_coords
        eye_coords = self.eye_video_click_coords
        self.main_video_click_coords = None
        self.eye_video_click_coords = None
        return main_coords, eye_coords
        
    def toggle_calibration(self):
        """Toggle calibration mode on and off"""
        # This will be connected to the controller's calibration mode
        calibration_active = getattr(self, 'calibration_active', False)
        calibration_active = not calibration_active
        
        if calibration_active:
            self.calibration_btn.configure(text="Stop Calibration")
            self.status_label.configure(text="Calibration started. Look at the target on screen.")
            # Disable other buttons during calibration
            self.select_eye_btn.configure(state="disabled")
            self.select_pupil_btn.configure(state="disabled")
        else:
            self.calibration_btn.configure(text="Start Calibration")
            self.status_label.configure(text="Calibration stopped.")
            # Re-enable buttons
            self.select_eye_btn.configure(state="normal")
            if getattr(self, 'eye_selected', False):
                self.select_pupil_btn.configure(state="normal")
                
        self.calibration_active = calibration_active
        # Return status to controller
        return calibration_active
        
    def update_calibration_target(self, position, point_num, total_points):
        """Update status with calibration information"""
        self.status_label.configure(text=f"Look at the target - Point {point_num+1}/{total_points}")
        return position
        
    def toggle_mouse_control(self):
        if hasattr(self, '_controller') and self._controller:
            print("GUI'den mouse kontrolü değiştiriliyor...")
            self._controller.toggle_mouse_control()
            # Buton metnini ve rengini güncelle
            if self._controller.mouse_control_active:
                self.mouse_control_btn.configure(
                    text="Mouse Kontrolünü Durdur",
                    fg_color="#c7503e",
                    hover_color="#a73e30"
                )
            else:
                self.mouse_control_btn.configure(
                    text="Mouse Kontrolünü Başlat",
                    fg_color="#3a7ebf",
                    hover_color="#306998"
                )
        else:
            print("HATA: Controller referansı bulunamadı!")
