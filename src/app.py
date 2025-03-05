from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import io

app = FastAPI(title="Skin Cancer Detection API")

# Load pre-trained model
model = tf.keras.models.load_model('models/skin_cancer_model.h5')

@app.post("/predict")
async def predict_skin_lesion(file: UploadFile = File(...)):
    """
    Predict skin lesion classification
    """
    # Read image
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess image
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img)[0][0]
    result = {
        'malignant_probability': float(prediction),
        'classification': 'Malignant' if prediction > 0.5 else 'Benign'
    }
    
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)