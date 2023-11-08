"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype("float32")


def load_mnist_dataset():
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    return train_images, test_images


def load_mnist_dataset_batch(train_size, batch_size, test_size):
    train_images, test_images = load_mnist_dataset()
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size)
        .batch(batch_size)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size)
        .batch(batch_size)
    )
    return train_dataset, test_dataset


@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: DictConfig):
    """Function to process the data"""
    train_images, test_images = load_mnist_dataset(
        config.data.train_size, config.data.batch_size, config.data.test_size
    )
    df = pd.DataFrame(train_images)
    df.to_csv(config.data.final)
    # tf.data.Dataset.zip(train_dataset, test_dataset).save(config.data.final)
    print(train_dataset, test_dataset)
    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")


# if __name__ == "__main__":
#    process_data()
