import cv2
import numpy as np
import time
import math
import dlib
import pyautogui

class CalibrationHelper:
    """
    Helper class for calibrating the eye tracker by collecting eye data at different screen positions.
    This helps improve tracking accuracy by mapping eye movement to screen coordinates.
    """
    
    def __init__(self):
        """Initialize the calibration helper."""
        self.calibration_data = {}
    
    def process_calibration(self, frame, controller):
        """
        Process calibration frame to record eye states at calibration points.
        
        Args:
            frame: Current camera frame
            controller: The MouseController instance
        """
        # Draw current calibration point on screen
        current_point = controller.calibration_points[controller.current_calibration_point]
        
        # Detect face and eyes
        face_coords = controller.face_detector.detect_face(frame)
        if face_coords is not None:
            face_rect = None
            if isinstance(face_coords, dlib.rectangle):
                face_rect = face_coords
            elif controller.face_detector.use_dlib is False and face_coords is not None:
                x, y, w, h = face_coords
                face_rect = dlib.rectangle(x, y, x + w, y + h)
                
            if face_rect:
                # Get eye landmarks
                left_eye_pts, right_eye_pts = controller.eye_tracker.get_eye_landmarks(frame, face_rect)
                
                # Measure eye openness
                left_eye_params = None
                right_eye_params = None
                
                if left_eye_pts is not None:
                    # Draw landmarks
                    for pt in left_eye_pts:
                        cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
                    
                    # Calculate eye parameters
                    left_eye_center = (np.mean(left_eye_pts[:, 0]).astype(int), 
                                      np.mean(left_eye_pts[:, 1]).astype(int))
                    
                    # Calculate eye openness (vertical distance between top and bottom points)
                    left_eye_height = np.max(left_eye_pts[:, 1]) - np.min(left_eye_pts[:, 1])
                    left_eye_width = np.max(left_eye_pts[:, 0]) - np.min(left_eye_pts[:, 0])
                    left_eye_aspect_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
                    
                    left_eye_params = {
                        'center': left_eye_center,
                        'height': left_eye_height,
                        'width': left_eye_width,
                        'aspect_ratio': left_eye_aspect_ratio
                    }
                
                if right_eye_pts is not None:
                    # Draw landmarks
                    for pt in right_eye_pts:
                        cv2.circle(frame, tuple(pt), 2, (0, 0, 255), -1)
                    
                    # Calculate eye parameters
                    right_eye_center = (np.mean(right_eye_pts[:, 0]).astype(int), 
                                       np.mean(right_eye_pts[:, 1]).astype(int))
                    
                    # Calculate eye openness (vertical distance between top and bottom points)
                    right_eye_height = np.max(right_eye_pts[:, 1]) - np.min(right_eye_pts[:, 1])
                    right_eye_width = np.max(right_eye_pts[:, 0]) - np.min(right_eye_pts[:, 0])
                    right_eye_aspect_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
                    
                    right_eye_params = {
                        'center': right_eye_center,
                        'height': right_eye_height,
                        'width': right_eye_width,
                        'aspect_ratio': right_eye_aspect_ratio
                    }
                
                # Record calibration data
                if left_eye_params and right_eye_params:
                    # Wait for user to focus on calibration point
                    if not hasattr(controller, 'calibration_timer'):
                        controller.calibration_timer = time.time()
                        
                    # Draw calibration target on frame with pulsating effect
                    target_radius = 20
                    elapsed = time.time() - controller.calibration_timer
                    pulse = abs(math.sin(elapsed * 3)) * 10
                    
                    # Draw concentric circles for better visibility
                    cv2.circle(frame, current_point, int(target_radius + pulse), (0, 165, 255), 2)
                    cv2.circle(frame, current_point, int(target_radius * 0.7 + pulse*0.7), (0, 165, 255), 2)
                    cv2.circle(frame, current_point, int(target_radius * 0.4 + pulse*0.4), (0, 165, 255), 2)
                    
                    # Label the calibration point
                    point_label = f"Point {controller.current_calibration_point + 1}/5"
                    cv2.putText(frame, point_label, 
                               (current_point[0] - 50, current_point[1] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Update GUI if method exists
                    if hasattr(controller.gui, 'update_calibration_target'):
                        controller.gui.update_calibration_target(current_point, 
                                                                controller.current_calibration_point, 
                                                                len(controller.calibration_points))
                    
                    # After 3 seconds, save data and move to next point
                    if time.time() - controller.calibration_timer > 3:
                        # Save calibration data
                        controller.calibration_data['points'].append({
                            'screen_pos': current_point,
                            'left_eye': left_eye_params,
                            'right_eye': right_eye_params
                        })
                        
                        # Move to next calibration point
                        controller.current_calibration_point += 1
                        
                        # Reset timer
                        if hasattr(controller, 'calibration_timer'):
                            delattr(controller, 'calibration_timer')
                        
                        # Check if calibration is complete
                        if controller.current_calibration_point >= len(controller.calibration_points):
                            controller.calibration_data['completed'] = True
                            if hasattr(controller.gui, 'status_label'):
                                controller.gui.status_label.configure(text="Calibration complete! Eye tracking optimized.")
                            controller.calibration_mode = False
                            if hasattr(controller.gui, 'calibration_active'):
                                controller.gui.calibration_active = False
                            if hasattr(controller.gui, 'calibration_btn'):
                                controller.gui.calibration_btn.configure(text="Start Calibration")
                            
                            # Enable normal selection again if button exists
                            if hasattr(controller.gui, 'select_eye_btn'):
                                controller.gui.select_eye_btn.configure(state="normal")
                            
                            # Process calibration data to optimize tracking
                            self.process_calibration_data(controller)
                        else:
                            # Move mouse to next calibration point
                            next_point = controller.calibration_points[controller.current_calibration_point]
                            pyautogui.moveTo(next_point[0], next_point[1])
        
        # Draw calibration status
        if hasattr(controller, 'calibration_timer'):
            elapsed = time.time() - controller.calibration_timer
            remaining = max(0, 3 - elapsed)
            cv2.putText(frame, f"Hold still: {remaining:.1f}s", 
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    def process_calibration_data(self, controller):
        """
        Process collected calibration data to optimize tracking parameters.
        Maps eye openness to vertical screen positions.
        
        Args:
            controller: The MouseController instance
        """
        if not controller.calibration_data['completed'] or len(controller.calibration_data['points']) < 5:
            return
        
        # Extract calibration info for upper and lower eye positions
        upper_points = controller.calibration_data['points'][0:2]  # Top left and top right
        lower_points = controller.calibration_data['points'][3:5]  # Bottom left and bottom right
        center_point = controller.calibration_data['points'][2]    # Center
        
        # Calculate average eye height/openness for upper and lower screen positions
        avg_upper_height_left = np.mean([p['left_eye']['height'] for p in upper_points if 'left_eye' in p])
        avg_lower_height_left = np.mean([p['left_eye']['height'] for p in lower_points if 'left_eye' in p])
        
        avg_upper_height_right = np.mean([p['right_eye']['height'] for p in upper_points if 'right_eye' in p])
        avg_lower_height_right = np.mean([p['right_eye']['height'] for p in lower_points if 'right_eye' in p])
        
        # Calculate horizontal eye positions for left/right screen positions
        left_points = [controller.calibration_data['points'][0], controller.calibration_data['points'][3]]  # Left top and bottom
        right_points = [controller.calibration_data['points'][1], controller.calibration_data['points'][4]]  # Right top and bottom
        
        avg_left_x_left_eye = np.mean([p['left_eye']['center'][0] for p in left_points if 'left_eye' in p])
        avg_right_x_left_eye = np.mean([p['left_eye']['center'][0] for p in right_points if 'left_eye' in p])
        
        avg_left_x_right_eye = np.mean([p['right_eye']['center'][0] for p in left_points if 'right_eye' in p])
        avg_right_x_right_eye = np.mean([p['right_eye']['center'][0] for p in right_points if 'right_eye' in p])
        
        # Store calibration parameters for eye tracker
        controller.eye_tracker.calibration = {
            'vertical_range': {
                'left': {
                    'upper': avg_upper_height_left,
                    'lower': avg_lower_height_left,
                    'center': center_point['left_eye']['height'] if 'left_eye' in center_point else None
                },
                'right': {
                    'upper': avg_upper_height_right,
                    'lower': avg_lower_height_right,
                    'center': center_point['right_eye']['height'] if 'right_eye' in center_point else None
                }
            },
            'horizontal_range': {
                'left': {
                    'left_pos': avg_left_x_left_eye,
                    'right_pos': avg_right_x_left_eye,
                    'center': center_point['left_eye']['center'][0] if 'left_eye' in center_point else None
                },
                'right': {
                    'left_pos': avg_left_x_right_eye, 
                    'right_pos': avg_right_x_right_eye,
                    'center': center_point['right_eye']['center'][0] if 'right_eye' in center_point else None
                }
            }
        }
        
        print("Calibration completed successfully!")
        print(f"Left eye vertical range: {avg_upper_height_left:.2f} (top) to {avg_lower_height_left:.2f} (bottom)")
        print(f"Right eye vertical range: {avg_upper_height_right:.2f} (top) to {avg_lower_height_right:.2f} (bottom)")