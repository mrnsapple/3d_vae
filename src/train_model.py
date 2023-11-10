"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import importlib
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from omegaconf import DictConfig


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    x = tf.cast(x, dtype=tf.float32)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


#@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def getModel(model_path):
    return importlib.import_module(model_path)


def save_image(predictions, image_path):
    for i in range(predictions.shape[0]):
        if i > 15:
            break
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(image_path)
    # plt.show()


def generate_and_save_images(model, epoch, test_sample, images_folder):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    plt.figure(figsize=(4, 4))
    save_image(predictions, "{}image_at_epoch_{:04d}.png".format(images_folder, epoch))


def display_image(epoch_no):
    return PIL.Image.open("image_at_epoch_{:04d}.png".format(epoch_no))


def perform_training(CVAE, train_dataset, test_dataset, batch_size, images_folder):
    optimizer = tf.keras.optimizers.Adam(1e-4)

    epochs = 10
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 2
    num_examples_to_generate = 16

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    tf.random.normal(shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)

    # Pick a sample of the test set for generating output images
    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch#[0:num_examples_to_generate, :, :, :]
    #save_image(test_sample, "{}base.png".format(images_folder))
    #generate_and_save_images(model, 0, test_sample, images_folder)

    for epoch in range(1, epochs + 1):
        print("Start epoch")
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        # display.clear_output(wait=False)
        print(
            "Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}".format(
                epoch, elbo, end_time - start_time
            )
        )
        #generate_and_save_images(model, epoch, test_sample, images_folder)
    return model

def get_vertices(dataset : pd.DataFrame, model_geo_batch:int):
    dataset : pd.DataFrame = dataset[dataset.columns[1:4]]
    vertices : np.array = dataset.to_numpy()
    print(vertices.shape)
    num_batches = int(vertices.size/(model_geo_batch*3))
    # Removed unused points, this is required to not get an error in
    # the following reshaping step
    vertices = vertices[:num_batches * model_geo_batch]
    vertices = vertices.reshape((num_batches, model_geo_batch, 3))
    return vertices

def train_test_split(dataset : np.array, train_percentage: float, model_geo_batch, batch_size: int):
    train_dataset  = dataset[:int(dataset.shape[0]*train_percentage)]
    test_dataset =  dataset[int(dataset.shape[0]*(train_percentage)):]
    print("train_dataset", train_dataset.shape)
    train_dataset = train_dataset.reshape((train_dataset.shape[0], model_geo_batch, 3, 1))
    test_dataset = test_dataset.reshape((test_dataset.shape[0], model_geo_batch, 3, 1))
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).batch(batch_size)
    return train_dataset, test_dataset

@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""
    dataset : pd.DataFrame = pd.read_csv(config.data.raw)
    x : np.array = get_vertices(dataset, config.data.model_geo_batch)
    print(x.shape)
    train_dataset, test_dataset = train_test_split(x, config.data.train_percentage, config.data.model_geo_batch, config.data.batch_size)
    #return
    model = getModel(config.model.name)
    model = perform_training(
        model.CVAE,
        train_dataset,
        test_dataset,
        config.data.batch_size,
        config.data.final_images,
    )
    model.save(config.model.checkpoint)

if __name__ == "__main__":
    train_model()
