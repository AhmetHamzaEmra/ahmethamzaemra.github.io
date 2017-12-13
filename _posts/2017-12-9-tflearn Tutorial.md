---
title:  "tflearn Tutorial"
date:   2017-12-09 21:24:23
published: true 
categories: [Deep Learning]
tags: [ML, Machine Learning, AI, Deep Learning]
---



<iframe src="http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html" width=100% heighy=500px></iframe>





TFLearn introduces a High-Level API that makes neural network building and training fast and easy. This API is intuitive and fully compatible with Tensorflow. I have find a lot of tutorial for populer modules such as keras or torch. On the other hand, there are not enough source for tflearn. Although documentation is great, some important functions are not explained fully. In this post I would like to give basic intuition about the library. Lets get started. 

****

## Intro to DNN

Tflearn provides basic structure just like keras. First we need to build the network architecture, then we can define and train the network. Since it is high level API, we dont need to initialize all weights and other parts, it does it for us but at the same time it does not provide freedom of tensorflow. Lets see basic neular network implementation. Throughout this post, I will be using mnsit dataset. 

### Traning 

``` python
net = tflearn.input_data(shape=[None, 784]) # input layer
net = tflearn.fully_connected(net, 64, activation = 'relu')		# hidden layer with relu activation 
net = tflearn.fully_connected(net, 10, activation='softmax') # output layer (softmax)
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(X, Y)
```

As we can see it is this easy to create basic three layer network. Great thing about tflearn, it provides variety of choices that we use for activation functions, loss fucntions, optimization algorithms, etc. I will get back to them later in the post.  I will not be explaining this fuctions and how they work, just how we can use it with tflearn. Now, lets see how to predict, save model, and load model. 

#### Layers:

* **Input_data **

Basicaly it is the input layer of the network. Important arguments that it takes, shape, data_preprocessing, and data_augmentation. the last two argument are optional also powerful.  We will get back to them. 

* **fully_connected**



* **regression** 



### PREDICTING  

After training the model, we can use `predict` to get predictions for test cases. 

```python
model.predict(X)
```

it will be returning the predicted probabilities. in order to get predict labels, we can use; 

```python
model.predict_label(X)
```





