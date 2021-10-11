import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons

from DSprites_VAE.src.model import VAE
from DSprites_VAE.src.utils import load_data, get_batch, create_categories_map


def load_model(batch_size=1, latent_dim=10, checkpoint_path='checkpoints/model1599'):
    vae_model = VAE(image_shape=(64, 64, 1), condition_shape=(1,), latent_dim=latent_dim, batch_size=batch_size)
    vae_model.load_weights(checkpoint_path)
    return vae_model


def activate_slider_widgets(model, z, c, im, fig):
    slider_positions = [0.1, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82]
    sliders = []
    for index, x in enumerate(slider_positions):
        ax_slider = plt.axes([x, 0.1, 0.0225, 0.25], facecolor='white')
        s = Slider(
            ax=ax_slider,
            label=r'$z_' + str(index) + '$',
            valmin=-10.0,
            valmax=10.0,
            valstep=0.0001,
            orientation="vertical",
            color='black',
        )
        s.set_val(float(z[:, index]))
        sliders.append(s)

    def update(_):

        for index, s in enumerate(sliders):
            z[:, index] = s.val
        prediction = model.decode(z, c, sigmoid=True)
        im.set_data(np.asarray(prediction).squeeze(0))
        fig.canvas.draw()

    for s in sliders:
        s.on_changed(update)

    return sliders


def initialize_plot(train_i, train_c, indices, model):
    plt.rcParams["figure.figsize"] = (7, 3)
    mpl.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()

    ax.margins(x=0)
    plt.axis('off')

    fig.suptitle('Disentangling the VAE latent space', fontsize=16)

    plt.subplots_adjust(left=0.1, bottom=0.455, right=0.84, top=0.757, wspace=0.05, hspace=0.05)

    x, c = get_batch([random.choice(indices)], train_i, train_c)
    mean, logvar = model.encode(x, c)
    z = model.reparameterize(mean, logvar)
    z = z.numpy()
    prediction = model.decode(z, c, sigmoid=True)
    im = ax.imshow(np.asarray(prediction).squeeze(0), cmap=plt.get_cmap('gray'))

    return c, z, im, fig


def show(checkpoint_path='checkpoints/model1299'):
    vae_model = load_model(checkpoint_path=checkpoint_path)
    shapes_map = {'Square': 0, 'Ellipse': 1, 'Heart': 2}
    train_images, train_categories = load_data()
    category_map = create_categories_map(train_categories)
    indices = category_map[shapes_map['Square']]
    random.shuffle(indices)

    c, z, im, fig = initialize_plot(train_i=train_images, train_c=train_categories, indices=indices, model=vae_model)
    sliders = activate_slider_widgets(model=vae_model, z=z, c=c, im=im, fig=fig)

    radio_ax = plt.axes([0.74, 0.5, 0.105, 0.2], facecolor='white')
    shapes_radio_button = RadioButtons(radio_ax, ('Square', 'Ellipse', 'Heart'))

    def shapefunc(val):
        indices = category_map[shapes_map[val]]
        random.shuffle(indices)

        x, c = get_batch(indices[0:1], train_images, train_categories)

        mean, logvar = vae_model.encode(x, c)
        z = vae_model.reparameterize(mean, logvar)
        z = z.numpy()
        # update sliders
        for index, s in enumerate(sliders):
            s.set_val(float(z[:, index]))
        im.set_data(np.asarray(vae_model.decode(z, c, sigmoid=True)).squeeze(0))
        fig.canvas.draw()

    shapes_radio_button.on_clicked(shapefunc)
    plt.show()


if __name__ == '__main__':
    show()
