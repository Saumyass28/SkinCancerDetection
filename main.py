import os
import sys
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_download import download_isic_dataset, prepare_dataset
from src.preprocessing import ImagePreprocessor
from src.model import SkinCancerClassifier
from src.evaluate import ModelEvaluator
from src.cloud_upload import LocalStorageUploader

def main():
    try:
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)

        # Download and prepare dataset
        print("Downloading and preparing dataset...")
        download_isic_dataset()
        prepare_dataset()

        # Preprocess images
        print("Preprocessing images...")
        preprocessor = ImagePreprocessor()
        X, y = preprocessor.prepare_dataset('data/raw')

        # Split dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training model...")
        classifier = SkinCancerClassifier()
        model, history = classifier.train(X_train, y_train, X_test, y_test)

        # Evaluate model
        print("Evaluating model...")
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        evaluator = ModelEvaluator(y_test, y_pred)
        metrics = evaluator.compute_metrics()
        evaluator.plot_confusion_matrix()
        evaluator.plot_roc_curve(model.predict(X_test))

        # Save detailed metrics to a JSON file
        metrics_path = 'data/processed/detailed_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'train_history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy']
                }
            }, f, indent=4)

        # Upload to local storage
        print("Uploading artifacts...")
        uploader = LocalStorageUploader()
        uploader.upload_project_artifacts()

        # Print final results
        print("\nSkin Cancer Detection Project Completed Successfully!")
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value}")

    except Exception as e:
        print(f"An error occurred during the project execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()