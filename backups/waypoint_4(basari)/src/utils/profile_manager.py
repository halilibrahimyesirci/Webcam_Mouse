import json
import os
import numpy as np
from pathlib import Path

class ProfileManager:
    def __init__(self):
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        
    def save_profile(self, name: str, calibration_data: dict, transformation_matrices: dict):
        """Kalibrasyon profilini kaydet"""
        # NumPy arrayları JSON formatına çevir
        serializable_data = {}
        for depth, data in calibration_data.items():
            serializable_data[str(depth)] = {
                'points': [p.tolist() if isinstance(p, np.ndarray) else p for p in data['points']],
                'gaze_points': [p.tolist() if isinstance(p, np.ndarray) else p for p in data['gaze_points']]
            }
            
        serializable_matrices = {}
        for depth, matrix in transformation_matrices.items():
            if matrix is not None:
                serializable_matrices[str(depth)] = matrix.tolist()
                
        profile_data = {
            "calibration_data": serializable_data,
            "transformation_matrices": serializable_matrices
        }
        
        file_path = self.profiles_dir / f"{name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=4)
            
    def load_profile(self, name: str) -> tuple:
        """Kalibrasyon profilini yükle"""
        file_path = self.profiles_dir / f"{name}.json"
        if not file_path.exists():
            return None, None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
            
        # JSON verilerini NumPy arraylere çevir
        calibration_data = {}
        for depth_str, data in profile_data["calibration_data"].items():
            depth = int(depth_str)
            calibration_data[depth] = {
                'points': [np.array(p) if isinstance(p, list) else p for p in data['points']],
                'gaze_points': [np.array(p) if isinstance(p, list) else p for p in data['gaze_points']]
            }
            
        transformation_matrices = {}
        for depth_str, matrix in profile_data["transformation_matrices"].items():
            depth = int(depth_str)
            transformation_matrices[depth] = np.array(matrix)
            
        return calibration_data, transformation_matrices
        
    def get_profiles(self) -> list:
        """Mevcut profil listesini döndür"""
        return [f.stem for f in self.profiles_dir.glob("*.json")]
        
    def delete_profile(self, name: str):
        """Profili sil"""
        file_path = self.profiles_dir / f"{name}.json"
        if file_path.exists():
            file_path.unlink()