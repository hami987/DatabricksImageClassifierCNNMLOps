import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
# Library for plotting the images and the loss function
import matplotlib.pyplot as plt
import pandas as pd
# We import the data set from tensorflow and build the model there
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def download_preprocess():

    # Download the data set
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_images, train_labels, test_images, test_labels, class_names

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels, class_names = download_preprocess()