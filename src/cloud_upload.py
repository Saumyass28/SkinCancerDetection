import os
import shutil
import json
from datetime import datetime

class LocalStorageUploader:
    def __init__(self, base_dir='project_artifacts'):
        """
        Initialize local storage uploader
        """
        # Create timestamp-based directory for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.storage_dir = os.path.join(base_dir, timestamp)
        
        # Create directories
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, 'visualizations'), exist_ok=True)
    
    def upload_file(self, file_path, destination_subpath=None):
        """
        Copy file to local storage directory
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        # If no specific destination subpath, use the original file name
        if destination_subpath is None:
            destination_subpath = os.path.basename(file_path)
        
        # Full destination path
        destination = os.path.join(self.storage_dir, destination_subpath)
        
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Copy the file
            shutil.copy2(file_path, destination)
            print(f"Copied {file_path} to {destination}")
        except Exception as e:
            print(f"Error copying file {file_path}: {e}")
    
    def upload_project_artifacts(self):
        """
        Upload project artifacts to local storage
        """
        artifacts = [
            'models/skin_cancer_model.h5',
            'models/quantized_model.tflite',
            'visualizations/confusion_matrix.png',
            'visualizations/roc_curve.png'
        ]
        
        # Create a summary JSON with run details
        summary = {
            'timestamp': datetime.now().isoformat(),
            'artifacts': []
        }
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                self.upload_file(artifact)
                summary['artifacts'].append(artifact)
        
        # Save summary JSON
        summary_path = os.path.join(self.storage_dir, 'run_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Artifacts stored in {self.storage_dir}")

if __name__ == "__main__":
    uploader = LocalStorageUploader()
    uploader.upload_project_artifacts()