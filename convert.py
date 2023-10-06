import onnx
import tensorflow as tf
import tf2onnx

# Load your TensorFlow model
model = tf.keras.models.load_model('your_model_path')

# Convert the TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Save the ONNX model to a file
onnx.save_model(onnx_model, 'your_output_model.onnx')
