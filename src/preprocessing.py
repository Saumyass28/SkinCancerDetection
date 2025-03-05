import os
import numpy as np
import cv2
import tensorflow as tf

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_and_preprocess_image(self, image_path):
        """
        Load, resize, and normalize image
        """
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)
            image = image / 255.0  # Normalize to [0,1]
            return image
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def prepare_dataset(self, data_dir):
        """
        Prepare dataset for model training
        """
        images = []
        labels = []
        
        for category in ['benign', 'malignant']:
            category_path = os.path.join(data_dir, category)
            label = 1 if category == 'malignant' else 0
            
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                image = self.load_and_preprocess_image(image_path)
                
                if image is not None:
                    images.append(image)
                    labels.append(label)
        
        # Ensure we have data
        if not images:
            raise ValueError("No images found in the dataset. Check data_download.py")
        
        return np.array(images), np.array(labels)

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    images, labels = preprocessor.prepare_dataset('data/raw')
    print(f"Loaded {len(images)} images")
    print(f"Label distribution: {np.bincount(labels)}")