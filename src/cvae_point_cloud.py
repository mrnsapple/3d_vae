import tensorflow as tf


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(60, 3, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=1, strides=(1, 1), activation="relu"
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=30 * 3 * 32, activation=tf.nn.relu, dtype='float32'),
                tf.keras.layers.Reshape(target_shape=(30, 3, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=[2,1],
                    padding="same",
                    activation="relu",
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )

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

"""
cvae = CVAE(1)
print(cvae.encoder.summary())
print(cvae.decoder.summary())

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 29, 1, 32)         320       
                                                                 
 conv2d_1 (Conv2D)           (None, 29, 1, 64)         2112      
                                                                 
 flatten (Flatten)           (None, 1856)              0         
                                                                 
 dense (Dense)               (None, 2)                 3714      
                                                                 
=================================================================
Total params: 6146 (24.01 KB)
Trainable params: 6146 (24.01 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_1 (Dense)             (None, 2880)              5760      
                                                                 
 reshape (Reshape)           (None, 30, 3, 32)         0         
                                                                 
 conv2d_transpose (Conv2DTr  (None, 30, 3, 64)         2112      
 anspose)                                                        
                                                                 
 conv2d_transpose_1 (Conv2D  (None, 60, 3, 32)         18464     
 Transpose)                                                      
                                                                 
 conv2d_transpose_2 (Conv2D  (None, 60, 3, 1)          289       
 Transpose)                                                      
                                                                 
=================================================================
Total params: 26625 (104.00 KB)
Trainable params: 26625 (104.00 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
"""