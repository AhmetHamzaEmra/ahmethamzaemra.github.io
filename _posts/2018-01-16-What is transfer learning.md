---
layout: post
title: "Transfer Learning with Tflearn"
tags: [Deep Learning, Transfer Learning]
<!-- feature-img: "assets/img/sample_feature_img.png" -->
---


In many case of computer vision task, amount of data and compatition power is problem. Ether we need thousand of data with labels (supervised learning case) or we need good GPUs, CPUs etc. to create good models. Unfortunately, not every one has that kind of data or computation available in their hand. But for cases like, Deep learning allows us to use models that was trained on other dataset such as imagenet (which contains 1.2 million images with 1000 categories) and transform this models in to something that we can use for our problem. This called transfer learning. There are diffent scenarios to do transfer learning. In this post, I would like to share one of them;

There method I want to explain is creating fixed feature extractor. So training a conv net is hard to do. Also finetuning requires some memory and computation. One way to fix this problem is, using this big pretrained network as feature extactor. Since we already know that lower layers of conv net gets lower level of features like edges etc and every layer on top of that get higher level of features. In most of the cases, this features that been extracted from other dataset can be useful in our dataset too. Ergo, insted of training network to find this features, we copy networks layers that already trained to find this features. But what will happen to last layer. For those layers we simply replace them with randomly initialized new layer and train it couple epochs. But how are we going to know how many layer we are going to replace? Well cs231n Course from Stanford Univesity give us a great tips for this issue. 

|                         |        Very Similar Dataset        |          Very Different Dataset          |
| :---------------------: | :--------------------------------: | :--------------------------------------: |
|  **Very little Data**   | Use linear classifier on top layer | You are in trouble... <br /> Try linear classifier from diffenet stages |
| **Quite a lot of data** |       Fine Tune a Few layers       |    Finetune a larger number of layers    |
|                         |                                    |                                          |

 So how do we implement all this. Since there are a lot of keras tutorial for transfer learning, I want to use [tflearn](http://tflearn.org/) and pretrained [vgg19 net](https://arxiv.org/pdf/1409.1556.pdf). I think it will be more usefull.

### implementation 

* **Pre process your data** 

VGG19 get inputs images in format of 224*224 in RGB. So this will be dimensions of our input layer. 

* **Load the network** 

in tflearn we need to write the structure and then we need to load the weights. All the code and weights can be found in this [link](https://github.com/AhmetHamzaEmra/tflearn_VGG19)

**Tips:** Since we are not going to use the last layers we can delete and use models as feature extractor. For this example I will only use one fully connected layer. So on forward propagation we are going to produce a tensor which contains all the features and then we wont be need to ever run this layers again. 

``` python
 
# Building 'VGG Network'
input_layer = input_data(shape=[None, 224, 224, 3])

block1_conv1 = conv_2d(input_layer, 64, 3, activation='relu', name='block1_conv1')
block1_conv2 = conv_2d(block1_conv1, 64, 3, activation='relu', name='block1_conv2')
block1_pool = max_pool_2d(block1_conv2, 2, strides=2, name = 'block1_pool')

block2_conv1 = conv_2d(block1_pool, 128, 3, activation='relu', name='block2_conv1')
block2_conv2 = conv_2d(block2_conv1, 128, 3, activation='relu', name='block2_conv2')
block2_pool = max_pool_2d(block2_conv2, 2, strides=2, name = 'block2_pool')

block3_conv1 = conv_2d(block2_pool, 256, 3, activation='relu', name='block3_conv1')
block3_conv2 = conv_2d(block3_conv1, 256, 3, activation='relu', name='block3_conv2')
block3_conv3 = conv_2d(block3_conv2, 256, 3, activation='relu', name='block3_conv3')
block3_conv4 = conv_2d(block3_conv3, 256, 3, activation='relu', name='block3_conv4')
block3_pool = max_pool_2d(block3_conv4, 2, strides=2, name = 'block3_pool')

block4_conv1 = conv_2d(block3_pool, 512, 3, activation='relu', name='block4_conv1')
block4_conv2 = conv_2d(block4_conv1, 512, 3, activation='relu', name='block4_conv2')
block4_conv3 = conv_2d(block4_conv2, 512, 3, activation='relu', name='block4_conv3')
block4_conv4 = conv_2d(block4_conv3, 512, 3, activation='relu', name='block4_conv4')
block4_pool = max_pool_2d(block4_conv4, 2, strides=2, name = 'block4_pool')

block5_conv1 = conv_2d(block4_pool, 512, 3, activation='relu', name='block5_conv1')
block5_conv2 = conv_2d(block5_conv1, 512, 3, activation='relu', name='block5_conv2')
block5_conv3 = conv_2d(block5_conv2, 512, 3, activation='relu', name='block5_conv3')
block5_conv4 = conv_2d(block5_conv3, 512, 3, activation='relu', name='block5_conv4')
block4_pool = max_pool_2d(block5_conv4, 2, strides=2, name = 'block4_pool')
flatten_layer = tflearn.layers.core.flatten (block4_pool, name='Flatten')

fc1 = fully_connected(flatten_layer, 4096, activation='relu')

# dp1 = dropout(fc1, 0.5)
# layer below this are not restored!
# fc2 = fully_connected(dp1, 4096, activation='relu', restore= False)
# dp2 = dropout(fc2, 0.5)
# network = fully_connected(dp2, num_classes, activation='softmax', restore=False)

regression = tflearn.regression(fc1, optimizer='adam',
loss='categorical_crossentropy',
learning_rate=0.001, restore=False)

model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
tensorboard_dir="./logs")

model.load("vgg19.tflearn", weights_only=True)

```

* **Produce the Features**

``` python
X_features = model.predict(X) # this will produce [num_example, 4096] matrix
# save the matrix for future  now you can use this features
# without going through all previous work
np.save("X_features", X_features)
```

* **Create model on top of features**

``` python
# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 4096])
dense1 = tflearn.fully_connected(input_layer, 4096, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 512, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, num_classes, activation='softmax')

sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
net = tflearn.regression(softmax, optimizer=sgd, 
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_features, Y, n_epoch=20, validation_set=0.1,
          show_metric=True, run_id="model")
```



This is all I want to share, For any commend or issue, you can connect me via email : hamza.emra [at] gmail.com 

Thanks