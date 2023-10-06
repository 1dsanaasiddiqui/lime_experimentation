import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert images to RGB format
train_images_rgb = np.stack((train_images,) * 3, axis=-1)
test_images_rgb = np.stack((test_images,) * 3, axis=-1)

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Flatten the images
train_images_flatten = train_images_rgb.reshape(train_images_rgb.shape[0], -1)
test_images_flatten = test_images_rgb.reshape(test_images_rgb.shape[0], -1)

# Build a simple feedforward neural network
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 3)))  # Input layer
model.add(Dense(128, activation='relu'))      # Hidden layer
model.add(Dense(10, activation='softmax'))    # Output layer

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images_rgb, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images_rgb, test_labels)
print(f'Test accuracy: {test_acc}')

# Save the Keras model to ONNX format
import onnxmltools
onnx_model = onnxmltools.convert_keras(model)
onnx_file_path = "model.onnx"
onnxmltools.utils.save_model(onnx_model, onnx_file_path)
