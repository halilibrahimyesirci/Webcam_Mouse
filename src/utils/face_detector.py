import cv2
import numpy as np
import os
import dlib  # Added dlib

class FaceDetector:
    def __init__(self, cascade_path=None, use_dlib=True):  # Added use_dlib flag and made cascade_path optional
        self.use_dlib = use_dlib
        if self.use_dlib:
            self.dlib_face_detector = dlib.get_frontal_face_detector()
        else:
            if not cascade_path or not os.path.exists(cascade_path):
                # Allow cascade_path to be None if use_dlib is True and cascade is not used
                if not self.use_dlib:
                    raise FileNotFoundError(f"Cascade file not found or not provided: {cascade_path}")
            if cascade_path:  # Only load if path is provided
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise ValueError(f"Failed to load cascade classifier: {cascade_path}")
            elif not self.use_dlib:  # If not using dlib and no cascade path, it's an error
                raise ValueError("Haar cascade path must be provided if not using dlib for face detection.")
        
    def detect_face(self, frame):
        if frame is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve face detection
        gray = cv2.equalizeHist(gray)  # Enhance contrast

        if self.use_dlib:
            faces_dlib = self.dlib_face_detector(gray, 0)  # Changed to 0 for faster processing, 1 for upsampling
            if len(faces_dlib) == 0:
                return None
            # Get the largest face (closest to camera) by area
            largest_face = max(faces_dlib, key=lambda rect: rect.width() * rect.height())
            return largest_face  # Return dlib.rectangle
        else:
            if not hasattr(self, 'face_cascade') or self.face_cascade.empty():
                print("Warning: Haar cascade for face detection is not loaded.")
                return None
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # Smaller scale factor for more accurate detection
                minNeighbors=8,   # Increased for more reliable detection
                minSize=(100, 100),  # Minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                return None
            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return largest_face  # Return (x, y, w, h)    
    def get_face_roi(self, frame, face_coords):
        if face_coords is None or frame is None:
            return None
            
        # Check if face_coords is a dlib rectangle
        if isinstance(face_coords, dlib.rectangle):
            x = face_coords.left()
            y = face_coords.top()
            w = face_coords.width()
            h = face_coords.height()
        else:
            # Assume it's a tuple or list of (x, y, w, h)
            x, y, w, h = face_coords
            
        return frame[y:y+h, x:x+w].copy()

    def draw_face(self, frame, face_coords):
        if face_coords is not None and frame is not None:
            # Check if face_coords is a dlib rectangle
            if isinstance(face_coords, dlib.rectangle):
                x = face_coords.left()
                y = face_coords.top()
                w = face_coords.width()
                h = face_coords.height()
            else:
                # Assume it's a tuple or list of (x, y, w, h)
                x, y, w, h = face_coords
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
