import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
from skimage.color import label2rgb

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert grayscale images to RGB format
train_images_rgb = np.stack((train_images,) * 3, axis=-1)
test_images_rgb = np.stack((test_images,) * 3, axis=-1)

# Normalize pixel values to be between 0 and 1
train_images_rgb, test_images_rgb = train_images_rgb / 255.0, test_images_rgb / 255.0

# Split the dataset for training and validation
train_images_rgb, val_images_rgb, train_labels, val_labels = train_test_split(train_images_rgb, train_labels, test_size=0.2, random_state=42)

# Check if the model is already saved, if not then train and save it
try:
    # Load the existing model
    model = models.load_model('my_mnist_model.h5')
except (OSError, IOError):
    # Build a simple feed-forward neural network
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images_rgb, train_labels, epochs=5, validation_data=(val_images_rgb, val_labels))

    # Save the model
    model.save('my_mnist_model.h5')

# Loop through all test images
for image_index in range(len(test_images)):
    # Choose an image for explanation
    image = test_images_rgb[image_index]
    label = test_labels[image_index]

    # Only generate explanations for images labeled as 7
    if label == 7:
        # Use LIME for explanations
        explainer = LimeImageExplainer()
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        explanation = explainer.explain_instance(
            image,
            predict_proba_fn,
            top_labels=1,
            num_features=5,
            num_samples=1000,
            segmentation_fn=segmenter,
        )

        # Display the original and superpixel-marked images
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False, min_weight=0.01)

        # Save the explanation as an image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
        ax1.set_title(f'Positive Regions for Label 7')

        # Display the original image
        ax2.imshow(image)
        ax2.set_title('Original Image')

        # Save the figure
        plt.savefig(f'marked_image_with_positive_regions_{image_index}.png', bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to prevent overlapping when generating multiple images

print("Explanations saved successfully.")
