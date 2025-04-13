import cv2
import mediapipe as mp
import numpy as np
import logging

# Configure logging to suppress TensorFlow warnings
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.camera = None
        self.camera_index = 0
        self.frame_width = None
        self.frame_height = None

        # MediaPipe iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]  # Sol göz bebeği landmarkları
        self.RIGHT_IRIS = [469, 470, 471, 472]  # Sağ göz bebeği landmarkları
        
        # Eye contour landmarks for relative position calculation
        self.LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def start_camera(self, camera_index=0):
        """Initialize and start the webcam"""
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)
        
        if self.camera.isOpened():
            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return True
        return False

    def stop_camera(self):
        """Stop and release the webcam"""
        if self.camera is not None:
            self.camera.release()

    def _calculate_relative_iris_position(self, landmarks, iris_points, contour_points):
        """Calculate iris position relative to eye contour"""
        # Get eye contour points
        contour = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                           for idx in contour_points])
        
        # Calculate eye bounding box
        min_x = np.min(contour[:, 0])
        max_x = np.max(contour[:, 0])
        min_y = np.min(contour[:, 1])
        max_y = np.max(contour[:, 1])
        
        # Get iris center
        iris = np.mean(np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                                for idx in iris_points]), axis=0)
        
        # Calculate relative position (0-1 range)
        rel_x = (iris[0] - min_x) / (max_x - min_x)
        rel_y = (iris[1] - min_y) / (max_y - min_y)
        
        return np.array([rel_x, rel_y])

    def _estimate_depth(self, face_landmarks):
        """Estimate depth from face size"""
        # Get points for face width calculation
        left_face = face_landmarks.landmark[234]  # Left face edge
        right_face = face_landmarks.landmark[454]  # Right face edge
        
        # Calculate face width in normalized coordinates
        face_width = abs(right_face.x - left_face.x)
        
        # Normalize depth to 0-1 range
        # Assuming typical face width ranges from 0.15 to 0.45 in normalized coordinates
        normalized_depth = np.clip((face_width - 0.15) / (0.45 - 0.15), 0, 1)
        
        return normalized_depth

    def get_eye_position(self, frame):
        """Process frame and return eye position"""
        if frame is None:
            return None

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Get image dimensions for MediaPipe
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate relative iris positions
        left_rel_pos = self._calculate_relative_iris_position(
            face_landmarks, self.LEFT_IRIS, self.LEFT_EYE_CONTOUR)
        right_rel_pos = self._calculate_relative_iris_position(
            face_landmarks, self.RIGHT_IRIS, self.RIGHT_EYE_CONTOUR)
        
        # Calculate depth
        depth = self._estimate_depth(face_landmarks)
        
        # Get iris points for visualization
        left_iris = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                             for idx in self.LEFT_IRIS])
        right_iris = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                              for idx in self.RIGHT_IRIS])
        
        # Average the positions from both eyes
        avg_pos = (left_rel_pos + right_rel_pos) / 2.0
        
        return {
            'left_eye': left_rel_pos,
            'right_eye': right_rel_pos,
            'gaze_point': avg_pos,
            'left_iris_points': left_iris,
            'right_iris_points': right_iris,
            'depth': depth,
            'original_frame': frame
        }