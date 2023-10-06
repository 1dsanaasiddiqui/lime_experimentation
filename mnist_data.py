import os
import gzip
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

def download_mnist(url, folder):
    # Create the folder if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Download MNIST dataset
    for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(f"{url}/{filename}", filepath)
        else:
            print(f"{filename} already exists.")

def extract_mnist(folder):
    # Extract and load MNIST dataset
    with gzip.open(os.path.join(folder, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        train_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(folder, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(os.path.join(folder, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        test_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(folder, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        test_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return train_images, train_labels, test_images, test_labels

def save_images(images, labels, folder):
    # Create the target directory if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Save images in the specified folder
    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(folder, f"{label}_{i}.png")
        image = Image.fromarray(image)
        image.save(image_path)

if __name__ == "__main__":
    url = "http://yann.lecun.com/exdb/mnist"
    folder = "mnist_dataset"

    # Download MNIST dataset
    download_mnist(url, folder)

    # Extract MNIST dataset
    train_images, train_labels, test_images, test_labels = extract_mnist(folder)

    # Save images to folders without subfolders
    save_images(train_images, train_labels, os.path.join(folder, "train"))
    save_images(test_images, test_labels, os.path.join(folder, "test"))
