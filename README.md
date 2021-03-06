# Disentangling the latent space of Variational Autoencoder
Disentangling the latent space learned by a VAE has been subject to a lot of research in recent years [1,2].
This repository provides a simple tensorflow implementation of a Variational Autoencoder and the objective function proposed by [1]:

![](equation.jpg)

Experiments showed that this modification to the objective function improves the tradeoff
between reconstruction quality and disentanglement.

## Usage
1. Run `pip install -r requirements.txt`
2. Either run `python visualize.py` to visualize an existing model or `python train.py` to train a new model.
3. The Dsprites dataset used in the repository can be found [here](https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz):

## Demo
Trained this model for 1200 epochs. <br>
<kbd><img src="https://github.com/Yoan-D/DisentangledVAE/blob/e17868f73fcdb2bac281daa2d141b774bd0e1323/demo.gif"/></kbd><br />

## References
[1] HIGGINS, I., MATTHEY, L., PAL, A., BURGESS, C., GLOROT, X., BOTVINICK, M., MOHAMED, S., AND LERCHNER, A. *beta-vae: Learning basic visual concepts with a constrained variational framework*. In ICLR (2017). <br>
[2] BURGESS, C. P., HIGGINS, I., PAL, A., MATTHEY, L., WATTERS, N., DESJARDINS, G., AND LERCHNER, A. *Understanding disentangling in β-VAE*. ArXiv abs/1804.03599 (2018).
