import numpy as np


def load_data(file='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'):
    dataset_zip = np.load(file, encoding='latin1')

    images = dataset_zip['imgs']
    latents_classes = dataset_zip['latents_classes']

    return images, latents_classes


def get_batch(indices, train_images, train_categories):
    shapes_as_categories = np.array([train_categories[i][1] for i in indices])
    images = np.array([train_images[i] for i in indices])

    return [images.reshape((images.shape[0], 64, 64, 1)).astype('float32'), shapes_as_categories.reshape(
        shapes_as_categories.shape[0], 1).astype('float32')]


def create_categories_map(train_categories):
    category_map = {}
    for index, c in enumerate(train_categories):
        if c[1] in category_map:
            category_map[c[1]].append(index)
        else:
            category_map[c[1]] = [index]

    return category_map
