import tensorflow as tf
from keras.src.engine import data_adapter
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv3D,
    Conv3DTranspose,
    Dense,
    Flatten,
    Input,
    Lambda,
    Reshape,
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2


class VAEModel(tf.keras.Model):
    def train_step(self, data):
        """The logic for one training step.

        Overriding the method to compute the loss considering that in vae we are not having a target variable
        But rather the original x is the target.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, x, y_pred, sample_weight)


class PrepareVae(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, input_shape, z_dim, learning_rate_1, momentum):
        super(PrepareVae, self).__init__()
        self.encoder = self.get_encoder(input_shape, z_dim)
        self.decoder = self.get_decoder(z_dim)
        dec_conv5 = self.decoder(self.encoder(self.encoder.inputs[0])[2])
        self.vae = VAEModel(self.encoder.inputs[0], dec_conv5)
        loss = self.compute_loss(self.encoder.inputs[0], dec_conv5)
        self.vae.add_loss(loss)
        sgd = SGD(learning_rate=learning_rate_1, momentum=momentum, nesterov=True)
        self.vae.compile(optimizer=sgd, metrics=["accuracy"])

    def sampling(self, args):
        mu, sigma = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + K.exp(0.5 * sigma) * epsilon

    def get_encoder(self, input_shape, z_dim):
        enc_in = Input(shape=input_shape)
        enc_conv1 = BatchNormalization()(
            Conv3D(
                filters=8,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="valid",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(enc_in)
        )
        enc_conv2 = BatchNormalization()(
            Conv3D(
                filters=16,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2),
                padding="same",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(enc_conv1)
        )
        enc_conv3 = BatchNormalization()(
            Conv3D(
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="valid",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(enc_conv2)
        )
        enc_conv4 = BatchNormalization()(
            Conv3D(
                filters=64,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2),
                padding="same",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(enc_conv3)
        )
        enc_fc1 = BatchNormalization()(
            Dense(units=343, kernel_initializer="glorot_normal", activation="elu")(
                Flatten()(enc_conv4)
            )
        )
        mu = BatchNormalization()(
            Dense(units=z_dim, kernel_initializer="glorot_normal", activation=None)(
                enc_fc1
            )
        )
        sigma = BatchNormalization()(
            Dense(units=z_dim, kernel_initializer="glorot_normal", activation=None)(
                enc_fc1
            )
        )
        z = Lambda(self.sampling, output_shape=(z_dim,))([mu, sigma])

        encoder = tf.keras.Model(enc_in, [mu, sigma, z])
        return encoder

    def get_decoder(self, z_dim):
        dec_in = Input(shape=(z_dim,))
        dec_fc1 = BatchNormalization()(
            Dense(units=343, kernel_initializer="glorot_normal", activation="elu")(
                dec_in
            )
        )
        dec_unflatten = Reshape(target_shape=(1, 7, 7, 7))(dec_fc1)

        dec_conv1 = BatchNormalization()(
            Conv3DTranspose(
                filters=64,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(dec_unflatten)
        )
        dec_conv2 = BatchNormalization()(
            Conv3DTranspose(
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2),
                padding="valid",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(dec_conv1)
        )
        dec_conv3 = BatchNormalization()(
            Conv3DTranspose(
                filters=16,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(dec_conv2)
        )
        dec_conv4 = BatchNormalization()(
            Conv3DTranspose(
                filters=8,
                kernel_size=(4, 4, 4),
                strides=(2, 2, 2),
                padding="valid",
                kernel_initializer="glorot_normal",
                activation="elu",
                data_format="channels_first",
            )(dec_conv3)
        )
        dec_conv5 = BatchNormalization(
            beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001)
        )(
            Conv3DTranspose(
                filters=1,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer="glorot_normal",
                data_format="channels_first",
            )(dec_conv4)
        )
        decoder = tf.keras.Model(dec_in, dec_conv5)
        return decoder

    def weighted_binary_crossentropy(self, target, output):
        loss = (
            -(
                98.0 * target * K.log(output)
                + 2.0 * (1.0 - target) * K.log(1.0 - output)
            )
            / 100.0
        )
        return loss

    def compute_loss(self, inputs, outputs):
        # kl_div = -0.5 * K.mean(1 + 2 * sigma - K.square(mu) - K.exp(2 * sigma))
        voxel_loss = K.cast(
            K.mean(
                self.weighted_binary_crossentropy(
                    inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7)
                )
            ),
            "float32",
        )  # + kl_div
        return voxel_loss

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
