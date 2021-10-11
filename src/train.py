import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from DSprites_VAE.src.loss import compute_loss
from DSprites_VAE.src.model import VAE
from DSprites_VAE.src.utils import load_data, get_batch


class ModelTrainer:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer

    @tf.function
    def train_step(self, model, batch, gamma, capacity):
        with tf.GradientTape() as tape:
            x, c = batch
            loss = compute_loss(model, x, c, gamma, capacity)
            tf.print('Total loss: ', loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def train(self, epochs=2000, batch_size=256, latent_dim=10, capacity_annealtime=200, capacity_start=150):
        train_images, train_categories = load_data()
        indices = list(range(train_images.shape[0]))
        random.shuffle(indices)
        vae_model = VAE(image_shape=(64, 64, 1), condition_shape=(1,), latent_dim=latent_dim, batch_size=batch_size)
        gamma_weight = tf.keras.backend.variable(1.0)
        capacity_weight = tf.keras.backend.variable(0.)

        total_batch = train_images.shape[0] // batch_size

        for epoch in range(epochs):
            start = time.time()

            for i in range(total_batch):
                batch_indices = indices[batch_size * i: batch_size * (i + 1)]
                batch = get_batch(batch_indices, train_images, train_categories)
                self.train_step(vae_model, batch, gamma_weight, capacity_weight)

            if epoch > capacity_start:
                new_weight = min(tf.keras.backend.get_value(capacity_weight) + (20. / capacity_annealtime), 20.)
                tf.keras.backend.set_value(capacity_weight, new_weight)
                tf.keras.backend.set_value(gamma_weight, 50)
                tf.print("Current gamma weight is {}".format(str(tf.keras.backend.get_value(gamma_weight))))
                tf.print(
                    "Current capacity weight is {}".format(str(tf.keras.backend.get_value(capacity_weight))))

            if (epoch + 1) % 100 == 0:
                vae_model.save_weights("checkpoints/model" + str(epoch))

            if (epoch + 1) % 1500 == 0:

                batch_indices = indices[batch_size * 1: batch_size * (1 + 1)]
                x, c = get_batch(batch_indices, train_images, train_categories)

                mean, logvar = vae_model.encode(x, c)
                z = vae_model.reparameterize(mean, logvar)
                predictions = vae_model.decode(z, c, sigmoid=True)

                for pred in predictions[:2]:
                    plt.imshow(np.asarray(pred).squeeze(-1), cmap='gray')
                plt.show()

            tf.print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def load_model(batch_size=25, latent_dim=10):
    train_images, train_categories = load_data()
    indices = list(range(train_images.shape[0]))
    random.shuffle(indices)
    batch_indices = indices[batch_size * 1: batch_size * (1 + 1)]
    x, c = get_batch(batch_indices, train_images, train_categories)

    vae_model = VAE(image_shape=(64, 64, 1), condition_shape=(1,), latent_dim=latent_dim, batch_size=batch_size)
    vae_model.load_weights('checkpoints/model1599')

    mean, logvar = vae_model.encode(x, c)
    z = vae_model.reparameterize(mean, logvar)
    predictions = vae_model.decode(z, c, sigmoid=True)

    for index, pred in enumerate(predictions):
        print(c[index])
        plt.imshow(np.asarray(pred).squeeze(-1), cmap='gray')
        plt.show()


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
    trainer = ModelTrainer(optimizer=opt)
    trainer.train()
