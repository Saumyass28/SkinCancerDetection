# AI-Powered Skin Cancer Detection System

## Project Overview
A deep learning system for classifying skin lesions as benign or malignant using computer vision and transfer learning with ResNet50V2.

## Features
- Automated skin lesion classification
- Transfer learning with ResNet50V2
- FastAPI REST endpoint for predictions
- Model quantization for efficient inference
- AWS S3 cloud integration
- Comprehensive model evaluation

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker
- AWS CLI (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/Saumyass28/skin-cancer-detection.git

# Install dependencies
pip install -r requirements.txt

# Download dataset
python src/data_download.py

# Preprocessing and Training
python src/preprocessing.py
python src/train.py

# Run FastAPI Server
python src/app.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t skin-cancer-detector .

# Run Docker container
docker run -p 8000:8000 skin-cancer-detector
```

## API Usage
```bash
# Predict skin lesion type
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

## Model Performance
- Accuracy: 92%
- Precision: 0.89
- Recall: 0.94
- F1 Score: 0.91

## Cloud Integration
AWS S3 upload script included for model and artifact storage.

## License
MIT License

## Contact
Saumya Sharma - saumyasharma281114@gmail.com
```