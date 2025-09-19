import tensorflow as tf

# -----------------------------
# Step 1: Load your Keras .h5 model
# -----------------------------
model = tf.keras.models.load_model("disease.h5")  # Replace with your model path

# -----------------------------
# Step 2: Setup TFLite converter for full INT8 quantization
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# -----------------------------
# Step 2a: Representative dataset generator (required for INT8)
# -----------------------------
def representative_dataset():
    for _ in range(100):
        # Replace shape with your model's input shape
        yield [tf.random.normal([1, 224, 224, 3])]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# -----------------------------
# Step 3: Convert and save the INT8 quantized model
# -----------------------------
tflite_quant_model = converter.convert()

OUTPUT_PATH = "disease_quant_int8.tflite"
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_quant_model)

print(f"✅ INT8 Quantized TFLite model created: {OUTPUT_PATH}")
