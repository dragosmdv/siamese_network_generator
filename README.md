# siamese_network_generator

## Notice

This is now an [OPEN Open Source](http://openopensource.org/) project. This python library contains the definition of a Siamese Network, ready to use for image predictions.

## Contents

* [Introduction](#introduction)
* [Usage](#usage)
* [Getting Started](#getting-started)
* [Examples](#examples)
* [Extensions](#extensions)

## Introduction


I used as base model a ResNet50 model pretrained on ImageNet and connected three more dense layers with 512 neurons on the first one and 256 on the last two. I set the target shape of input to (200,200,3). This model is designed to generate embedding for images, so I set the trainable attribute to true only for the layers that belonged to the last convolution block of the ResNet architecture, and freeze the other layers. This setup resulted in 60,510,720 Trainable params and 14,657,920 non-trainable params. Hereinafter we will call this model the Embedding Model and will note the embedding of an input image A with f(A).

Having the Embedding Model, I defined the Siamese Network which receives a triplet of images formed from an anchor, a positive and a negative, generates the
embeddings and returns two distances: between the anchor and the positive embeddings, and between the anchor and the negative embeddings. The anchor and the positive being the same class of images and the negative a different one. To compute those distances, I used a custom layer class named DistanceLayer, and for the distance formula we used the Squared Euclidian Distance:

<p align="center">
    d^2(p, q) = (p1 − q1)^2 + (p2 − q2)^2 + ... + (pn − qn)^2
</p>

Also represented as:

<p align="center">
  d^2(p, q) = ||p − q||^2
</p>

The final step was to define the Siamese Model with a custom training and testing loop. This model used the Siamese Network to compute the Triplet Loss Function:

<p align="center">
  L(A, P, N) = max(||f(A) − F(P)||^2 − ||f(A) − f(N)||^2 + margin, 0)
</p>

Where ’margin’ represent the actual margin that is enforced between positive and negative pairs. This loss function is used to ensure that an image A (anchor) is closer to all other images P (positive) of the same class than it is to any image N (negative). A visual representation of the effect of this function is the following image

![My Image](tripletLoss.jpg)

Now the Siamese Network is ready for training. To test the results of the network, I generated the embedding of all the images from the training set, then I took all the test images one by one, generated an embedding and find the closest matching embedding from the training set. For the training of such a model, it is required to
select triplets of anchor, positive and negative images.

## Usage

Given a dataset of images from multiple classes where each class must contain one or more images, this library can be used to train a model to recognize the class of a new image by selecting the closest match from the initial train dataset.

The library contains two functions:

**1. generate_model(test_file, train_file, save_path, epochs)**
- **test_file** represents the path to your test dataset package. The package must contain a sub-folder for each individual class, having the name of the class equal to the name of the sub-folder. In each sub-folder there must be similar images from the same class.
- **train_file** represents the path to your train dataset package. Same as the first package, it must contain a sub-folder for each individual class, having the name of the class equal to the name of the sub-folder. In each sub-folder there must be similar images from the same class. The number of samples from the two datasets could vary, it is recommended a ratio of 7/3 train/test. 
- **save_path** represents the path where you want yo save your created model
- **epochs** represents the number of epochs that you want to use in training your model

**2. test_prediction(image_path, model_path, train_dataset_path)**
- **image_path** represents the path of the image that you want to predict
- **model_path** represents the path of the generated model (identical to **save_path** if you want to use the previousely generated model)
- **train_dataset_path** represents the path to a dataset that contains the classes from which that you want to predict. In order to obtain good results, the initial train dataset is used most of the time.

## Getting started

In order to use the presented model, the user is required to have already installed Python 3.5 or higher.

There are two ways to use the library:
1. Download the repository and copy the files in a python project
2. Download and install the compressed python library: Download the file located at dist/siamese_network_generator-0.1.0-py3-none-any.whl in this repository, then in your python project you can run the following comand (replacing "/path/to" with your actual path):

```bash
$ pip install /path/to/siamese_network_generator-0.1.0-py3-none-any.whl
```

Once you have installed your Python library, you can import it using:
```bash
import siamese_network_generator
from siamese_network_generator import simese_network_actions
```

For both ways, you need to install the some dependencies for the library to run properly. You can easily install these dependencies using PIP, but note that they might take some time. Note to update PIP before using the following commands:

```bash
pip install tensorflow==2.5
pip install pandas
pip install keras

```

You can also run this library using Anaconda, to train the model with your gpu. For this case, you will need to run the following commands in your environment:
```bash
conda install -c conda-forge tensorflow==2.5
conda install pandas
conda install -c conda-forge keras

```



## Examples

To ilustrate an example a simple usage of this library is presented in the test folder.

First, we set the train_file, test_file, save_path and epochs variables

```bash
    test_file = "datasets/test_images"
    train_file = "datasets/train_images"
    save_path = "generated_models/myModel_v16"
    epochs = 1
```

Then, using the **generate_model** function, we generate a model in the "generated_models/" package of the current folder.

If everything works properly, you should see in logs a "Fetching triplet for ..." statement for each class, first for train images then for test  images. Then you sould see a loading bar that describes the progress of the training for an epoch:
```bash
    14/14 [==============================] - 112s 7s/step - loss: 0.6285 - val_loss: 0.1865
```

After the model is saved, we can use it to predict some images. For the first test, we used a image from the class named "Opera_Națională_Română_din_Cluj-Napoca" that was not included in the train process. This image is saved in "test_image.jpg". After running the prediction, in the console will appear the name of the predicted class, among with its tensor that contains the specific distance (that varies depending on the training of the model)

```bash
('Opera_Națională_Română_din_Cluj-Napoca', <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.9060962], dtype=float32)>)
```

For the second example, we used a image from the class named "Biserica_evanghelică_din_Cluj-Napoca" that **was** included in the train dataset. The model was trained to recognize the exact same image, so it should recognize the same class name with a distance of 0

```bash
('Biserica_evanghelică_din_Cluj-Napoca', <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>)
```

## Extensions

This library contains a simple implementation of a siamese network for image predictions. The model can easily be integrated in both web or mobile applications.

The network is very customizable in terms of model params. Feel free to experiment the alterations of the model with different base models, architectures, learning params and datasets.

