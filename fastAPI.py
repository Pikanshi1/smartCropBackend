
import os
import warnings

# -----------------------------
# Force CPU and suppress TF warnings
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import gdown
import tensorflow as tf

# -----------------------------
# Suppress TensorFlow warnings
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# -----------------------------
# Set model path
# -----------------------------
if os.name == "nt":  # Windows
    MODEL_PATH = "disease.tflite"
else:  # Linux / Render
    MODEL_PATH = "/tmp/disease.tflite"

print(f"Model path set to: {MODEL_PATH}")

# -----------------------------
# Google Drive model configuration
# -----------------------------
FILE_ID = "1AesQxhc6UsZoPm3JCY4VRsbkdVwtWx3n"  # replace with your TFLite model file ID
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
# Load TFLite model
# -----------------------------
try:
    print("🔄 Loading the TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Plant Disease Prediction API (TFLite)")

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running (TFLite)"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        return {"error": "Model not loaded."}
    try:
        # Preprocess image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((64, 64))  # match model input
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# -----------------------------
# Run server (local only)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "fastAPI:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
