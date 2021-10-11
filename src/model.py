import tensorflow as tf


class VAE(tf.keras.Model):

    def __init__(self, image_shape, condition_shape, latent_dim, batch_size):
        super(VAE, self).__init__()
        self.image_shape = image_shape
        self.condition_shape = condition_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        encoder_input = tf.keras.layers.Input(shape=self.image_shape)
        condition = tf.keras.layers.Input(shape=self.condition_shape)
        c = tf.keras.layers.Embedding(input_dim=3, output_dim=10)(condition)
        c = tf.keras.layers.Dense(self.image_shape[0] * self.image_shape[1])(c)
        c = tf.keras.layers.Reshape((self.image_shape[0], self.image_shape[1], 1))(c)

        x = tf.keras.layers.Concatenate(axis=-1)([encoder_input, c])
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Flatten()(x)

        mean = tf.keras.layers.Dense(self.latent_dim, activation='linear', kernel_initializer='zeros', )(x)
        logvar = tf.keras.layers.Dense(self.latent_dim, activation='linear', kernel_initializer='zeros', )(x)

        self.encoder = tf.keras.Model([encoder_input, condition], [mean, logvar])
        self.encoder.summary()

        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        condition = tf.keras.layers.Input(shape=self.condition_shape)

        x = tf.keras.layers.Dense(units=8 * 8 * 64, activation=tf.nn.relu)(decoder_input)
        x = tf.keras.layers.Reshape(target_shape=(8, 8, 64))(x)

        c = tf.keras.layers.Embedding(input_dim=3, output_dim=10)(condition)
        c = tf.keras.layers.Dense(8 * 8)(c)
        c = tf.keras.layers.Reshape((8, 8, 1))(c)

        x = tf.keras.layers.Concatenate(axis=-1)([x, c])
        x = tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=3, strides=2, padding='same',
            activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding='same',
            activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2, padding='same',
            activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)

        output = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same')(x)

        self.decoder = tf.keras.Model([decoder_input, condition], output)
        self.decoder.summary()

    @tf.function
    def sample(self, c, eps=None):
        if eps is None:
            eps = tf.keras.backend.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                                 stddev=1.0)
        return self.decode(eps, c, sigmoid=True)

    def encode(self, x, c):
        mean, logvar = self.encoder([x, c])
        return mean, logvar

    def reparameterize(self, mean, logvar):
        epsilon_std = 1.0

        epsilon = tf.keras.backend.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                                 stddev=epsilon_std)
        return mean + tf.keras.backend.exp(logvar / 2) * epsilon

    def decode(self, z, c, sigmoid=False):
        logits = self.decoder([z, c])
        if sigmoid:
            logits = tf.nn.sigmoid(logits)
        return logits
