import os

# -----------------------------
# Set model path depending on OS
# -----------------------------
if os.name == "nt":  # Windows
    MODEL_PATH = "disease.h5"
else:  # Linux / Render
    MODEL_PATH = "/tmp/disease.h5"

print(f"Model path set to: {MODEL_PATH}")

# Force CPU and suppress TF warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# -----------------------------
# Google Drive model configuration
# -----------------------------
FILE_ID = "1AesQxhc6UsZoPm3JCY4VRsbkdVwtWx3n"  # your Google Drive file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_model(url, path):
    """Download model from Google Drive only if it does not exist."""
    if not os.path.exists(path):
        try:
            print("📥 Downloading model from Google Drive...")
            gdown.download(url, path, quiet=False)
            if os.path.exists(path):
                print(f"✅ Download completed: {path}")
            else:
                print("❌ Download failed: File not found after download.")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
    else:
        print(f"✅ Model already exists: {path}")

# Download model if needed
download_model(URL, MODEL_PATH)

# -----------------------------
# Load the model
# -----------------------------
try:
    print("🔄 Loading the model...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]  # replace with your actual class names

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Plant Disease Prediction API")

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}

    # Open uploaded image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((128, 128))  # match model input size

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

# -----------------------------
# Run server (for local testing)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "fastAPI:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )
