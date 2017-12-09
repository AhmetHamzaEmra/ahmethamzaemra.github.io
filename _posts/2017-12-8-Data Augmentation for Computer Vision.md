---
title:  "Data Augmentation for Computer Vision!"
date:   2017-12-08 15:04:23
published: true 
categories: [Deep Learning]
tags: [ML, Machine Learning, AI, OpenCV, Deep Learning]
---



In general, working with image data is fun. But it also has some disadvantages in Deep learning.  One of the biggest problem in the Deep learning is it requires a lot of training example. In this post, I want to share some tips to increase the number of training example for computer vision tasks. 



First Lets get some intuition about basic computer vision tasks and opencv.  Opencv is open source computer vision module. It is one of the easiest way to get working on images. in this post I will be using Opencv Python api.



Note: 

Installation might give a headache. For mac or linux users `pip3 install opencv-python` is enought for now. Some complex tasks it might require, I recommend to follow the installation tutorial, but for this post it is not necessary. 



## OpenCV

We will be using this red panda img through out. the image is 580 by 636, colored image. 



<img src="/images/post4/redpanda.jpg">



First lets go over some functions:

1. reading image from pc

```python
>>> import cv2  # importing the module 
>>> img = cv2.imread("redpanda.jpg") # reading img from file
>>> img.shape
(580, 636, 3)
```

As we can see, we have 580 by 636 image with 3 color changes. In opencv first channel represents blue, second green and third red. In other modules such as matplotlib, images are in RGB format. So when we want to display it, we get strange images like the following one.

<img src="/images/post4/BGR.jpg">

 but we can change it with one function. 

2. Color Conversion

cvtColor fucntion takes two argument, one of them is the image we want to change, other one is what color to what color we want to conver

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
```

<img src="/images/post4/RBG.jpg">

Also since we are trying to fit this images to deep learning model, using colored images mean working with three times more parameter for input layer. For some problems, the color is important. For example, classifying diffent cats. Some cat might have just same structure but diffent colors. So in that case it's necessary to use all three chanels. But lets say we need to classify dogs with cats. Since color is not important we can convert the images to gray scale with using same function. 

```python
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # conver RGB to Gray
```

<img src="/images/post4/GRAY.jpg">



3. Resizing 

Another important thing is resizing. Deep learning models require to have same shape for every training example. But not all images are on same size, some of them are HD some of them are looks like they been taken by toaster. So to get same size for every image, We can use opencv's resize function as follows.

```python
resized = cv2.resize(img, (50,50)) # resize image to 50 by 50
```

<img src="/images/post4/resized.jpg">



## Data Augmentation 

1. **Fliping**

Fliping the images is one of the way to get some extra images. It is very handy tool but  also it has some disadvantages with it. For instance, we are classifying the CIFAR 10 dataset and we want to increase trainig set size. If we do horizontal flip, since we are adding upside down cars, horses, etc. this process might confuse our model. Ergo, it is great tool as long as we are careful. Lets see an example:

``` python
# the function takes two argument, first one image, 
# second one orientation we want to flip
horizontal_flip = cv2.flip( img, 0 )
vertical_img = cv2.flip( img, 1 )
both_img = cv2.flip( img, -1 )
```

<img src="/images/post4/flip.jpg">

2. Rotating 

Rotating is another way to increase images in the dataset and also it has the same disadvantage as flipping! 

```python
# -30 is the rotation angle
matrix = cv2.getRotationMatrix2D((height/2, width/2), -30, 1)
rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
```

<img src="/images/post4/rotated.jpg">

Tip: insted of rotating every image with same angle, randomly rotating would generate more general distribution.

3. Bluring

Some times bluring the images also helps. On the other hand if the detailes is important in the image, it is not recommended. 

``` python
# (5,5) vertical and horizontal blur
# must be odd number!
blur = cv2.GaussianBlur(img, (55,55), 0)
```

<img src="/images/post4/blur.jpg">



This is all for this post. See you next time! 