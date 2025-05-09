import customtkinter as ctk
import time
import numpy as np
from PIL import Image, ImageDraw

class CalibrationWindow(ctk.CTkToplevel):
    def __init__(self, parent, calibrator, eye_tracker, on_complete):
        super().__init__(parent)
        
        self.calibrator = calibrator
        self.eye_tracker = eye_tracker
        self.on_complete = on_complete
        
        # Set fullscreen
        self.attributes('-fullscreen', True)
        self.configure(fg_color="black")
        
        # Initialize variables
        self.current_point_index = 0
        self.current_depth_phase = 0  # 0: yakın, 1: orta, 2: uzak
        self.depth_positions = [
            {"name": "YAKIN"},
            {"name": "ORTA"},
            {"name": "UZAK"}
        ]
        
        self.points = self.calibrator.get_calibration_points()
        self.collection_time = 2.0
        self.collected_data = []
        self.point_start_time = None
        
        # Create canvas
        self.canvas = ctk.CTkCanvas(
            self,
            highlightthickness=0,
            bg="black"
        )
        self.canvas.pack(expand=True, fill="both")
        
        # Initial instructions
        self.show_initial_instructions()
        
        # Bind keys
        self.bind("<Escape>", lambda e: self.end_calibration())
        self.bind("<Key>", self.start_calibration)

    def show_initial_instructions(self):
        """Show initial calibration instructions"""
        instructions = (
            "3 Aşamalı Kalibrasyon\n\n"
            "1. YAKIN: Kameraya yakın mesafeden bakın\n"
            "2. ORTA: Normal oturuş mesafesinden bakın\n"
            "3. UZAK: Kameradan uzak mesafeden bakın\n\n"
            "Her aşama için:\n"
            "- Ekrandaki noktaları takip edin\n"
            "- Her noktada 2 saniye bekleyin\n"
            "- Tüm noktalar tamamlandığında yeni mesafe için hazır olun\n\n"
            "YAKIN mesafe kalibrasyonu için herhangi bir tuşa basın"
        )
        
        self.canvas.create_text(
            self.winfo_screenwidth() // 2,
            self.winfo_screenheight() // 2,
            text=instructions,
            fill="white",
            font=("Helvetica", 20),
            justify="center"
        )

    def collect_point_data(self):
        """Collect eye tracking data for current point"""
        if self.point_start_time is None:
            self.point_start_time = time.time()
        
        elapsed_time = time.time() - self.point_start_time
        
        if elapsed_time < self.collection_time:
            ret, frame = self.eye_tracker.camera.read()
            if ret:
                tracking_result = self.eye_tracker.get_eye_position(frame)
                if tracking_result and 'gaze_point' in tracking_result:
                    self.collected_data.append(tracking_result['gaze_point'])
            
            self.after(10, self.collect_point_data)
        else:
            self.process_point_data()

    def process_point_data(self):
        """Process collected gaze data and move to next point"""
        if self.collected_data:
            avg_gaze = np.mean(self.collected_data, axis=0)
            self.calibrator.add_calibration_point(
                self.points[self.current_point_index],
                avg_gaze,
                self.current_depth_phase
            )
            
            # Move to next point
            self.current_point_index += 1
            self.collected_data = []
            self.point_start_time = None
            
            # If all points are done for current depth
            if self.current_point_index >= len(self.points):
                self.current_point_index = 0
                self.current_depth_phase += 1
                
                # If all depths are done
                if self.current_depth_phase >= len(self.depth_positions):
                    self.complete_calibration()
                else:
                    self.show_depth_phase_instructions()
            else:
                self.after(500, self.show_next_point)
        else:
            self.after(500, self.show_next_point)

    def show_depth_phase_instructions(self):
        """Show instructions for the next depth phase"""
        self.canvas.delete("all")
        next_depth = self.depth_positions[self.current_depth_phase]
        
        instructions = (
            f"{next_depth['name']} Mesafe Kalibrasyonu\n\n"
            f"Lütfen kameradan {next_depth['name'].lower()} mesafeye geçin\n\n"
            "Hazır olduğunuzda herhangi bir tuşa basın"
        )
        
        self.canvas.create_text(
            self.winfo_screenwidth() // 2,
            self.winfo_screenheight() // 2,
            text=instructions,
            fill="white",
            font=("Helvetica", 24),
            justify="center"
        )
        
        # Rebind key event for next phase
        self.bind("<Key>", self.start_next_depth_phase)

    def start_next_depth_phase(self, event=None):
        """Start calibration for next depth phase"""
        self.bind("<Key>", lambda e: None)  # Disable key binding
        self.after(500, self.show_next_point)

    def show_next_point(self):
        """Display the next calibration point"""
        # Clear canvas
        self.canvas.delete("all")
        
        # Get current point coordinates
        x, y = self.points[self.current_point_index]
        
        # Draw point
        point_size = 20
        outer_size = 40
        
        # Draw outer circle (guide)
        self.canvas.create_oval(
            x - outer_size//2, y - outer_size//2,
            x + outer_size//2, y + outer_size//2,
            outline="gray"
        )
        
        # Draw point
        self.canvas.create_oval(
            x - point_size//2, y - point_size//2,
            x + point_size//2, y + point_size//2,
            fill="white"
        )
        
        # Show current depth phase info
        depth_info = f"{self.depth_positions[self.current_depth_phase]['name']} mesafe kalibrasyonu"
        self.canvas.create_text(
            self.winfo_screenwidth() // 2,
            50,
            text=depth_info,
            fill="white",
            font=("Helvetica", 16)
        )
        
        # Start collecting data
        self.collect_point_data()

    def start_calibration(self, event=None):
        """Start the initial calibration sequence"""
        self.bind("<Key>", lambda e: None)  # Disable key binding
        self.after(500, self.show_next_point)

    def complete_calibration(self):
        """Complete the calibration process"""
        if self.calibrator.calculate_transformation():
            print("Kalibrasyon başarılı!")
        else:
            print("Kalibrasyon başarısız - yeterli veri toplanamadı")
        
        self.end_calibration()

    def end_calibration(self):
        """Clean up and close calibration window"""
        if self.on_complete:
            self.on_complete()
        self.destroy()