# Product Categorization for Inventory Management

The task is to categorize the different products based on the image dataset given. (Image Classification).

I took the dataset from a HackerEarth competition - Deep Learning Challenge 1. It'll be automatically downloaded in the notebook.

## Methodology:

### Pre-processing:

1. Image Resize - 256x256px
2. Center Cropping
3. Normalization

### Training with resnet34 architecture:

**Training - 1 (Parameters)**

1. ```batch_size = 64```
2. ```learning_rate = 0.01```
3. ```epochs = 5```

Accuracy - 65%

#### Choosing Learning Rate

**Cyclic Learning Rates**: Simply keep increasing the learning rate from a very small value, until the loss starts decreasing. We can plot the learning rate across batches to see what this looks like.

```learning_rate = 0.01```

### Improving Model

#### Data Augmentation

Randomly changing the images in ways that shouldn't impact their interpretation, such as horizontal flipping, zooming, and rotating.

**Training - 2 (Parameters)**

1. ```learning_rate = 0.01```
2. ```learn.precompute = False``` (Calculates new weights, discarding the ones trained from 1st training)
3. ```cycle_len = 2``` ([Stochastic gradient descent with restarts](https://arxiv.org/abs/1608.03983))

Accuracy - 69%

#### Differential learning rate annealing

```cycle_mult = 2```

The other layers have *already* been trained to recognize imagenet photos (whereas our final layers were randomly initialized), so we want to be careful of not destroying the carefully tuned weights that are already there.

Basically, the earlier layers (layers in the beginning) have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason we will use different learning rates for different layers: the first few layers will be at 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 1e-2 as before.



Final Accuracy - 83.9%



## Requirements:

The project is done in "pytorch". The source code for the library "[fastai](https://github.com/fastai/fastai)" (primarily used) must be forked/cloned to run the notebook and reproduce the results. Place my ipynb notebook inside ./courses/dl1 of the "fastai" repo.

Though the library is in its beta development, it's very stable for deploying deep learning models built on torch.

**I'd highly recommend contributing to its development.**

Note: AWS EC2 instances were used for GPU. The processing will be very slow on CPU.



 