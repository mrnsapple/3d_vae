"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
import numpy as np
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


def load_obj(fn):
    fin = open(fn, "r")
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f


def export_obj(out, v, f):
    with open(out, "w") as fout:
        for i in range(v.shape[0]):
            fout.write("v %f %f %f\n" % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write("f %d %d %d\n" % (f[i, 0], f[i, 1], f[i, 2]))


@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: DictConfig):
    """Function to process the data"""
    # tf.data.Dataset.zip(train_dataset, test_dataset).save(config.data.final)
    v, f = load_obj(config.data.obj)
    print(v, f)
    export_obj("/home/oriol/tools/3d_vae/data/obj/testout.obj", v, np.array())
    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == "__main__":
    process_data()
