## Written Summary

For this exercise I built a simple GAN that generates handwritten digits from the MNIST dataset.

The generator takes random noise as input and tries to turn it into a digit image, while the discriminator tries to tell real MNIST images from fake ones. I kept the model simple and used small fully connected networks, binary cross-entropy loss, and Adam. During training I saved sample images from the same fixed noise every few epochs to track how the generator changed over time.

The results were decent. At the start the outputs were just noise, but after some training they became clearly more digit-like. The final samples were not perfect, but many of them looked like recognizable handwritten numbers, so the model was definitely learning something real.

The main difficulty was that GAN training is less straightforward than normal supervised training. The loss curves are harder to interpret, and the image quality can improve unevenly. If I were improving this, I would probably try a better architecture, tune the hyperparameters more, and maybe test a convolutional GAN or a Wasserstein GAN.
