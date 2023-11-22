"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import importlib

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from utils import npytar


def getModel(model_path):
    return importlib.import_module(model_path)


def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr


def data_loader(fname, input_shape):
    reader = npytar.NpyTarReader(fname)
    xc = np.zeros((reader.length(),) + input_shape, dtype=np.float32)
    reader.reopen()
    for ix, (x, _name) in enumerate(reader):
        xc[ix] = x.astype(np.float32)
    return 3.0 * xc - 1.0


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""
    model = getModel(config.model.name)
    batch_size = config.model.batch_size
    epoch_num = config.model.epoch_num
    prepare_vae = model.PrepareVae(
        config.model.input_shape,
        config.model.z_dim,
        config.model.learning_rate_1,
        config.model.momentum,
    )
    encoder = prepare_vae.encoder
    decoder = prepare_vae.decoder
    vae = prepare_vae.vae
    plot_model(
        encoder,
        to_file="{}/vae_encoder.pdf".format(config.data.model_plots),
        show_shapes=True,
    )
    plot_model(
        decoder,
        to_file="{}/vae_decoder.pdf".format(config.data.model_plots),
        show_shapes=True,
    )
    plot_model(
        vae, to_file="{}/vae.pdf".format(config.data.model_plots), show_shapes=True
    )
    tf.debugging.disable_traceback_filtering()
    data_train = data_loader(
        "/home/oriol/tools/3D-VAE/datasets/shapenet10_chairs_nr.tar",
        config.model.input_shape,
    )
    vae.fit(
        data_train,
        epochs=epoch_num,
        batch_size=batch_size,
        validation_data=(data_train, None),
        callbacks=[LearningRateScheduler(learning_rate_scheduler)],
    )
    vae.save_weights(config.model.checkpoint)


@hydra.main(config_path="../config", config_name="main", version_base=None)
def test_model(config: DictConfig):
    model = tf.keras.models.load_model(config.model.checkpoint)
    dataset: pd.DataFrame = pd.read_csv(config.data.raw)
    x: np.array = get_vertices(dataset, config.data.model_geo_batch)
    _, test_dataset = train_test_split(
        x,
        config.data.train_percentage,
        config.data.model_geo_batch,
        config.model.batch_size,
    )
    for test_x in test_dataset:
        mean, logvar = model.encode(test_x)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        test_x = tf.reshape(
            test_x,
            [test_x.shape[0] * test_x.shape[1], test_x.shape[2], test_x.shape[3]],
        )
        predictions = tf.reshape(
            predictions,
            [
                predictions.shape[0] * predictions.shape[1],
                predictions.shape[2],
                predictions.shape[3],
            ],
        )
        print("Exporting objs")
        utils.export_obj(config.data.original_obj, test_x, np.array([]))
        utils.export_obj(config.data.created_obj, predictions, np.array([]))


if __name__ == "__main__":
    train_model()
    # test_model()
