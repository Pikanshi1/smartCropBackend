from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("disease.h5")

# Class names (replace with actual class names)
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))   # match model input size

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence
    }

# ✅ Run server
if __name__ == "__main__":
    uvicorn.run("fastAPI:app", host="127.0.0.1", port=8000, reload=True)
