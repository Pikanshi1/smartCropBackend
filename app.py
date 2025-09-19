import os
import time
import warnings
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import gdown

# -----------------------------
# Force CPU and suppress TF warnings
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# -----------------------------
# TFLite model path
# -----------------------------
TFLITE_PATH = "/tmp/disease_quant_int8.tflite"  # For Render / Linux
if os.name == "nt":
    TFLITE_PATH = os.path.join(os.getcwd(), "disease_quant_int8.tflite")

# -----------------------------
# Google Drive download link
# -----------------------------
FILE_ID = "1oHvQPLlkVXN9cNGGfEWcStlaaNN3X5T4"  # Your INT8 TFLite file ID
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# -----------------------------
# Download model if not exists
# -----------------------------
if not os.path.exists(TFLITE_PATH):
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Downloading TFLite INT8 model (Attempt {attempt+1})...")
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
# Load TFLite INT8 model
# -----------------------------
try:
    print("🔄 Loading TFLite INT8 model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite INT8 model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Error loading TFLite INT8 model: {e}")

# -----------------------------
# Class names (replace with real classes)
# -----------------------------
CLASS_NAMES = [f"Class_{i}" for i in range(38)]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="🌱 Plant Disease Prediction API (TFLite INT8)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        shape = input_details[0]['shape']
        image = Image.open(file.file).convert("RGB")
        image = image.resize((shape[2], shape[1]))
        img_array = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)  # INT8 input

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_class = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        class_name = CLASS_NAMES[pred_class]

        return {
            "predicted_class": pred_class,
            "class_name": class_name,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# -----------------------------
# Debug endpoint to see raw probabilities
# -----------------------------
@app.post("/predict_debug", response_class=JSONResponse)
async def predict_debug(file: UploadFile = File(...)):
    """
    Returns raw output probabilities for all classes.
    Useful to debug why model predicts same class.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        shape = input_details[0]['shape']
        image = Image.open(file.file).convert("RGB")
        image = image.resize((shape[2], shape[1]))
        img_array = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        return {
            "raw_output": output_data.tolist(),  # Probabilities / logits for all classes
            "predicted_class": int(np.argmax(output_data)),
            "class_name": CLASS_NAMES[int(np.argmax(output_data))],
            "confidence": float(np.max(output_data))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True  # Set to False on Render deployment
    )
