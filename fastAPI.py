import os
import warnings
# -----------------------------
# Force CPU and suppress TF warnings
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# -----------------------------
# Set model path
# -----------------------------
if os.name == "nt":  # Windows
    MODEL_PATH = "disease.h5"
else:  # Linux / Render
    MODEL_PATH = "/tmp/disease.h5"

print(f"Model path set to: {MODEL_PATH}")

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
    if not os.path.exists(path):
        print("📥 Downloading model from Google Drive...")
        gdown.download(url, path, quiet=False)
        if os.path.exists(path):
            print(f"✅ Download completed: {path}")
        else:
            print("❌ Download failed.")
    else:
        print(f"✅ Model already exists: {path}")

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
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

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
    try:
        image = Image.open(file.file).convert("RGB")
        image = image.resize((128, 128))  # match model input size
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array[np.newaxis, ...]

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# -----------------------------
# Run server (only for local use)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "fastAPI:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
        # ⚠️ Removed reload=True for Render
    )
