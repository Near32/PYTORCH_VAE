# Disentangled Variational AutoEncoders with PyTorch

This repository is an attempt at replicating some results presented in Irina Higgins et al.'s papers :

*	["Early Visual Concept Learning with Unsupervised Deep Learning"](https://arxiv.org/pdf/1606.05579.pdf).
*	["beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](https://openreview.net/forum?id=Sy2fzU9gl)

## Requirements 

### ["dSprites - Disentanglement testing Sprites dataset"](https://github.com/deepmind/dsprites-dataset) :

In order to use DeepMind's ["dSprites - Disentanglement testing Sprites dataset"](https://github.com/deepmind/dsprites-dataset), you need to clone their repository and place it at the root of this one.

```
git clone https://github.com/deepmind/dsprites-dataset.git
```

### XYS-latent dataset :

In order to use the XYS-latent dataset, you need to :

1. download it [here](https://www.dropbox.com/s/luexrritqj4hv5r/dataset-XYS-latent.tar.gz?dl=0)
2. extract it at the root of this repository's folder.


## Experiments

### XYS-latent dataset :

Using this dataset and the following hyperparameters :

* Number of latent variables : 10
* learning rate : 1e-5
* "Temperature" hyperparameter Beta : 5e3
* Number of layers of the decoder : 5
* Base depth of the convolution/deconvolution layers : 32
* Stacked architecture : [x]

Real images : ![real1](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/real_images.png)

Considering one column, every three row contains :

1. Full image.
2. Right-eye patch extracted from the full image.
3. Left-eye patch extracted from the full image.


Epoch | Reconstruction | Latent Space 
------|---------------|---------------
1 | ![reconst1-1](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/reconst_images/1.png) | ![gen1-1](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/gen_images/1.png)
10 | ![reconst1-10](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/reconst_images/10.png) | ![gen1-10](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/gen_images/10.png)
30 | ![reconst1-30](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/reconst_images/30.png) | ![gen1-30](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/gen_images/30.png)
70 | ![reconst1-70](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/reconst_images/70.png) | ![gen1-70](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/gen_images/70.png)
100 | ![reconst1-100](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/reconst_images/100.png) | ![gen1-100](/doc/XYS-latent/test--XYS--img256-lr1e-05-beta5000.0-layers5-z10-conv32-stacked/gen_images/100.png)


#### Observations :

The S-scale latent variable seems to have been clearly disentangled while the other two latent variables, X and Y coordinates of the gaze on the camera plane, seem to be requiring a finer level of details from the decoder to show good reconstructions. Further analysis show that those latent variables are also quite nicely disentangled eventhough it is difficult to see here.
 
## Disclaimers

I do not own any rights on some of the datasets that have been used and experienced with, namely :

*	DeepMind's ["dSprites - Disentanglement testing Sprites dataset"](https://github.com/deepmind/dsprites-dataset).

*	Yann Lecun's ["MNIST"](http://yann.lecun.com/exdb/mnist/), through its [PyTorch's Dataset](http://pytorch.org/docs/master/torchvision/datasets.html#mnist) wrapper.


