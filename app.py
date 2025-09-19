import os
import warnings
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import tensorflow as tf
import time

# -----------------------------
# Force CPU and suppress TF warnings
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# -----------------------------
# Model path depending on platform
# -----------------------------
if os.name == "nt":  # Windows local
    TFLITE_PATH = os.path.join(os.getcwd(), "disease.tflite")
else:  # Linux / Render
    TFLITE_PATH = "/tmp/disease.tflite"

# -----------------------------
# Google Drive TFLite file ID
# -----------------------------
FILE_ID = "1gVr_7OuZi5Of7wb0YZRfg76K3_47qJCS"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# -----------------------------
# Download TFLite model if not exists
# -----------------------------
if not os.path.exists(TFLITE_PATH):
    import gdown
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Downloading TFLite model from Google Drive (Attempt {attempt+1})...")
            gdown.download(DRIVE_URL, TFLITE_PATH, quiet=False)
            if os.path.exists(TFLITE_PATH):
                print(f"✅ Model downloaded successfully to {TFLITE_PATH}")
                break
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if attempt < max_retries - 1:
                print("⏳ Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise RuntimeError("Failed to download TFLite model after multiple attempts.")

# -----------------------------
# Load TFLite model
# -----------------------------
try:
    print("🔄 Loading the TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Error loading TFLite model: {e}")

# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="🌱 Plant Disease Prediction API (TFLite)")

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Predict endpoint (JSON response)
# -----------------------------
@app.post("/predict", response_class=JSONResponse, summary="Predict plant disease from leaf image")
async def predict(
    file: UploadFile = File(..., description="Upload a leaf image (.jpg or .png) via form-data")
):
    """
    Accepts a file via form-data (not JSON) and returns prediction as JSON:
    {
        "predicted_class": int,
        "class_name": str,
        "confidence": float
    }
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    try:
        # Preprocess image
        input_shape = input_details[0]['shape']
        target_height, target_width = input_shape[1], input_shape[2]

        image = Image.open(file.file).convert("RGB")
        image = image.resize((target_width, target_height))
        img_array = np.expand_dims(np.array(image, dtype=np.float32)/255.0, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        class_name = CLASS_NAMES[predicted_class]

        return {
            "predicted_class": predicted_class,
            "class_name": class_name,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True  # Use reload=True locally, set False on Render
    )
