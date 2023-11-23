"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import importlib

import hydra
import tensorflow as tf
from data_retrieval import data_loader
from omegaconf import DictConfig
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model


def getModel(model_path):
    return importlib.import_module(model_path)


def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = 0.005
    return lr


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
        config.data.raw,
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


if __name__ == "__main__":
    train_model()
