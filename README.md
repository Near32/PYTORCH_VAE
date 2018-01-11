# Disentangled Variational AutoEncoders with PyTorch

This repository is an attempt at replicating some results presented in Irina Higgins et al.'s papers :

*	["Early Visual Concept Learning with Unsupervised Deep Learning"](https://arxiv.org/pdf/1606.05579.pdf).
*	["beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](https://openreview.net/forum?id=Sy2fzU9gl)

## Requirements 

In order to use DeepMind's ["dSprites - Disentanglement testing Sprites dataset"](https://github.com/deepmind/dsprites-dataset), you need to clone their repository and place it at the root of this one.

```
git clone https://github.com/deepmind/dsprites-dataset.git
```
## Disclaimers

The datasets that have been used and experienced with are :

*	DeepMind's ["dSprites - Disentanglement testing Sprites dataset"](https://github.com/deepmind/dsprites-dataset).

*	Yann Lecun's ["MNIST"](http://yann.lecun.com/exdb/mnist/), through its [PyTorch's Dataset](http://pytorch.org/docs/master/torchvision/datasets.html#mnist) wrapper.


