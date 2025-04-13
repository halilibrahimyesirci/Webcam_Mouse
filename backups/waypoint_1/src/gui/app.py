import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
from pynput import keyboard
from tracking.eye_tracker import EyeTracker
from calibration.calibrator import Calibrator
from utils.mouse_controller import MouseController
from gui.calibration_window import CalibrationWindow

class EyeTrackingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Eye Tracking Mouse Control")
        self.geometry("1024x768")
        
        # Initialize components
        self.eye_tracker = EyeTracker()
        self.calibrator = Calibrator()
        self.mouse_controller = MouseController()
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
        self.smoothing_label = ctk.CTkLabel(
            self.controls_frame,
            text="Mouse Smoothing:"
        )
        self.smoothing_label.pack(padx=10, pady=(10, 0))
        
        self.smoothing_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=self._update_smoothing
        )
        self.smoothing_slider.set(0.3)
        self.smoothing_slider.pack(padx=10, pady=(0, 10))
        
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
        self.mouse_controller.set_smoothing(float(value))
        
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
                    
                    # Convert normalized coordinates to pixel coordinates
                    left_x = int(left_eye[0] * frame.shape[1])
                    left_y = int(left_eye[1] * frame.shape[0])
                    right_x = int(right_eye[0] * frame.shape[1])
                    right_y = int(right_eye[1] * frame.shape[0])
                    
                    # Draw points for each eye
                    cv2.circle(frame, (left_x, left_y), 5, (255, 0, 0), -1)  # Blue for left eye
                    cv2.circle(frame, (right_x, right_y), 5, (0, 0, 255), -1)  # Red for right eye
                    
                    # Calculate and draw oval between eyes
                    center_x = (left_x + right_x) // 2
                    center_y = (left_y + right_y) // 2
                    
                    # Calculate oval parameters
                    dx = abs(right_x - left_x)
                    dy = abs(right_y - left_y)
                    
                    # Draw oval
                    axes = (dx, max(dy, dx//2))  # Width and height of oval
                    angle = np.degrees(np.arctan2(right_y - left_y, right_x - left_x))
                    cv2.ellipse(frame, (center_x, center_y), axes, angle, 0, 360, (0, 255, 255), 2)
                    
                    # Draw center point (actual mouse position)
                    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)  # Green
                    
                    # Update mouse position if control is enabled
                    if self.mouse_control_enabled:
                        screen_pos = self.calibrator.get_screen_position(gaze_point)
                        if screen_pos is not None:
                            self.mouse_controller.move_mouse(*screen_pos)
                            # Update debug canvas
                            self._update_debug_visualization(screen_pos)
                
                # Convert frame to CTkImage for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                
                # Update label
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo

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