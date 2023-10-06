# import onnxruntime
# import numpy as np
# from lime.lime_image import LimeImageExplainer
# from PIL import Image
# from skimage.segmentation import quickshift
# from lime.wrappers.scikit_image import SegmentationAlgorithm




# onnx_model_path = 'model.onnx'
# providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
# session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
# input_size = (28, 28)

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize(input_size)
#     image = np.array(image).astype(np.float32)
#     image /= 255.0
#     # Convert grayscale to fake RGB by duplicating the single channel
#     image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image


# def predict_fn(images):
#     # images is a numpy array with shape (batch_size, height, width)
    
#     # Preprocess each image and make predictions
#     predictions = []
#     for image in images:
#         preprocessed_image = preprocess_image(image)
        
#         # Run prediction using ONNX model
#         input_name = session.get_inputs()[0].name
#         output_name = session.get_outputs()[0].name
#         result = session.run([output_name], {input_name: preprocessed_image})
        
#         predictions.append(result[0])

#     return np.array(predictions)

# image_path = 'mnist_dataset/test/0_1001.png'
# image = preprocess_image(image_path)

# # Create a LIME explainer
# explainer = LimeImageExplainer()

# # Define a segmentation function using quickshift
# def segmentation_fn(image):
#     # Check if the image is grayscale
#     if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
#         # Convert the grayscale image to RGB-like by duplicating the single channel
#         image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
#     # print(image.shape)
#     return quickshift(image, kernel_size=1, max_dist=200, ratio=0.2)


# # Explain the prediction for the image
# # explanation = explainer.explain_instance(
# #     image,  # Extract the single image from the batch
# #     classifier_fn=predict_fn,
# #     top_labels=1,
# #     hide_color=0,
# #     num_samples=1000,
# #     segmentation_fn=segmentation_fn,
# # )

# segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)


# explanation = explainer.explain_instance(image, 
#                                          classifier_fn = predict_fn, 
#                                          top_labels=1, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

# import onnxruntime
# import numpy as np
# from lime.lime_image import LimeImageExplainer
# from PIL import Image
# from skimage.segmentation import quickshift
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# from tensorflow.keras.datasets import mnist

# # Load the MNIST dataset
# (train_images, _), (_, _) = mnist.load_data()

# # Convert grayscale images to RGB format
# train_images_rgb = np.stack((train_images,) * 3, axis=-1)

# onnx_model_path = 'model.onnx'
# providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
# session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
# input_size = (28, 28)

# # Use the first image from the MNIST dataset converted to RGB
# image = np.expand_dims(train_images_rgb[0], axis=0)

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize(input_size)
    
#     # Convert grayscale image to RGB format
#     image_rgb = np.stack((np.array(image),) * 3, axis=-1)

#     image_rgb = image_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
#     image_rgb = np.expand_dims(image_rgb, axis=0)  # Add batch dimension
#     return image_rgb


# # Modify this according to your actual output format
# # Assume your model is a single-label model
# def predict_proba_fn(images):
#     # images is a numpy array with shape (batch_size, height, width, channels)
#     # Preprocess each image and make probability predictions
#     probabilities = []
#     for image in images:
#         # Convert the image data type to float
#         image = image.astype(np.float32)

#         # Add an extra dimension to the image
#         image = np.expand_dims(image, axis=0)

#         # Run probability prediction using ONNX model
#         input_name = session.get_inputs()[0].name
#         output_name = session.get_outputs()[0].name
#         result = session.run([output_name], {input_name: image})

#         # Assuming result[0] is the class probabilities output
#         # For a single-label model, set the probability for the actual class to 1, others to 0
#         label_index = np.argmax(result[0])
#         prob = np.zeros_like(result[0])
#         prob[0, label_index] = 1.0

#         probabilities.append(prob)

#     return np.array(probabilities)





# # Create a LIME explainer
# explainer = LimeImageExplainer()

# # Define a segmentation function using quickshift
# def segmentation_fn(image):
#     return quickshift(image, kernel_size=1, max_dist=200, ratio=0.2)

# # Explain the prediction for the image
# segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
# explanation = explainer.explain_instance(
#     image[0],
#     classifier_fn=predict_proba_fn,
#     top_labels=1,
#     hide_color=0,
#     num_samples=1000,
#     segmentation_fn=segmenter,
#     num_features=5
# )

# # Display the original and superpixel-marked images
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# Image.fromarray(mark_boundaries(temp / 2 + 0.5, mask)).show(title='Superpixel-Marked Image')
# original_image = Image.fromarray((image[0] * 255).astype(np.uint8))
# original_image.show(title='Original Image')

import onnxruntime
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from PIL import Image
from skimage.segmentation import quickshift
from lime.wrappers.scikit_image import SegmentationAlgorithm
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, _), (_, _) = mnist.load_data()

# Convert grayscale images to RGB format
train_images_rgb = np.stack((train_images,) * 3, axis=-1)

onnx_model_path = 'model.onnx'
providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
input_size = (28, 28)

# Use the first 10 images from the MNIST dataset converted to RGB
images = train_images_rgb[:10]

def predict_proba_fn(images):
    # images is a numpy array with shape (batch_size, height, width, channels)
    # Preprocess each image and make probability predictions
    probabilities = []
    for img in images:
        # Convert the image data type to float
        img = img.astype(np.float32)

        # Add an extra dimension to the image
        img = np.expand_dims(img, axis=0)

        # Run probability prediction using ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img})

        probabilities.append(result[0])

    return np.array(probabilities)

# Flatten and concatenate the images for LimeTabularExplainer
flattened_images = images.reshape((images.shape[0], -1))

# Create a LIME explainer for tabular data
explainer = LimeTabularExplainer(training_data=flattened_images, feature_names=None, mode="classification")

# Explain the predictions for the images
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
for i in range(len(images)):
    explanation = explainer.explain_instance(
        data_row=flattened_images[i],
        classifier_fn=predict_proba_fn,
        top_labels=1,
        num_features=5,
        num_samples=1000,
        segmentation_fn=segmenter,
    )
    print(f"Explanation for image {i + 1}: {explanation.local_exp}")
