import matplotlib.pyplot as plt
import pandas as pd
from data.preprocessing import download_preprocess

def plot_10(train_images, class_names, train_labels):
    # Show the first 10 images
    plt.figure(figsize=(10,10))
    for i in range(10):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # Die CIFAR Labels sind Arrays, deshalb benötigen wir den extra Index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels, class_names = download_preprocess()
    plot_10(train_images, class_names, train_labels)