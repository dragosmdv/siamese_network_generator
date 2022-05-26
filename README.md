# siamese_network_generator

Introduction

This library contains the definition of a Siamese Network, ready to use for image predictions

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

![alt text]([http://url/to/img.png](https://i.imgur.com/ynewXQM.jpeg))

Now the Siamese Network is ready for training. To test the results of the network, I generated the embedding of all the images from the training set, then I took all the test images one by one, generated an embedding and find the closest matching embedding from the training set. For the training of such a model, it is required to
select triplets of anchor, positive and negative images.
