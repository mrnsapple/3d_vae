import importlib

import hydra
from data_retrieval import data_loader
from omegaconf import DictConfig
from utils import save_volume


def getModel(model_path):
    return importlib.import_module(model_path)


@hydra.main(config_path="../config", config_name="main", version_base=None)
def test_model(config: DictConfig):
    """Function to train the model"""
    model = getModel(config.model.name)
    data_train = data_loader(
        config.data.raw,
        config.model.input_shape,
    )
    print(data_train.shape)
    print(data_train[0, 0, :].shape)
    prepare_vae = model.PrepareVae(
        config.model.input_shape,
        config.model.z_dim,
        config.model.learning_rate_1,
        config.model.momentum,
    )
    vae = prepare_vae.vae
    vae.load_weights(config.model.checkpoint)
    reconstructions = vae.predict(data_train)
    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0
    for i in range(reconstructions.shape[0]):
        if i > 10:
            break
        save_volume.save_output(
            reconstructions[0, 0, :], 32, "{}".format(config.data.final), i
        )
        mult_rec = reconstructions[i] * reconstructions[i + 2]
        save_volume.save_output(
            mult_rec[0, :], 32, "{}/multip".format(config.data.final), i
        )
        add_rec = reconstructions[i] + reconstructions[i + 2]
        save_volume.save_output(
            add_rec[0, :], 32, "{}/add_rec".format(config.data.final), i
        )
        min_rec = reconstructions[i] - reconstructions[i + 2]
        save_volume.save_output(
            min_rec[0, :], 32, "{}/min_rec".format(config.data.final), i
        )

    # for i in range(data_train.shape[0]):

    #    save_volume.save_output(data_train[i, 0, :], 32, config.data.processed, i)


if __name__ == "__main__":
    test_model()
