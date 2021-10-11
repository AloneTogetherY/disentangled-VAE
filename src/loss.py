import tensorflow as tf


def compute_loss(model, x, c, gamma_weight, capacity_weight):
    mean, logvar = model.encode(x, c)
    z = model.reparameterize(mean, logvar)
    reconstruction = model.decode(z, c)

    total_reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x,
                                                                        logits=reconstruction)
    total_reconstruction_loss = tf.reduce_sum(total_reconstruction_loss, 1)

    kl_loss = 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5

    total_loss = tf.reduce_mean(total_reconstruction_loss * 3 + (
            gamma_weight * tf.abs(kl_loss - capacity_weight)))
    return total_loss
