---
layout: post
title: "Intelegence lock"
tags: [Deep Learning, Artificial Intelengece]
---



As Andrej Karpathy said, Software2.0 is creating revalatrion in the field of software engineering. Now AI  is taking most of the heavy lifting in complexity and giving incredible results in the industry. One of this application is locks with face recognition. Although it sounds simple to combine these two technologies, it is more complex than most other applications. 



First of all, we have a problem with a dataset. Which is it is very hard to collect more than a couple of pictures of every person that we need to give access. If we can get a couple hundred pictures per person, One can easily train a simple model to do classification. But let us say we have 1000 employees that we want to give access. we need to get ~1.000.000 employee pictures and couple hundred pictures from other people in the World so the model can know if unknown people to lock to the door. On top of this, if we hire anybody else, We would need to train again. Screw that, we need another method. So we need a method that we can just use one picture per person to classify all of them and doesn't need to train when someone changes. That method is Landmark technique. Let's mark specific points in the face. Then we can use this transformation to classify people. Since everybody will have a different distance between two eyes or eyes and notice (except the twins!) we can tread this new measurement as embedding and if they are too close to each other in the multidimensional space we can they are the same person, vice versa.

Let's see how can we implement this face landmarks model. Lets start with finding a dataset. If you google it you will find tons of dataset ([like this one](https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points)) Then you just train a neural network to map faces to landmark mesurements. Lets see a simple implementation with keras. Our goal is to find and mark 15 face landmarks for this dataset. Here is some examples from dataset.

![](../images/post5/1.png)

```python
# load the dataset and clean a bit 
face_images_db = np.load('face_images.npz')['face_images']
facial_keypoints_df = pd.read_csv('facial_keypoints.csv')

numMissingKeypoints = facial_keypoints_df.isnull().sum(axis=1)
allKeypointsPresentInds = np.nonzero(numMissingKeypoints == 0)[0]

faceImagesDB = face_images_db[:,:,allKeypointsPresentInds]
facialKeypointsDF = facial_keypoints_df.iloc[allKeypointsPresentInds,:].reset_index(drop=True)

(imHeight, imWidth, numImages) = faceImagesDB.shape
numKeypoints = facialKeypointsDF.shape[1] / 2

print('number of remaining images = %d' %(numImages))
print('image dimentions = (%d,%d)' %(imHeight,imWidth))
print('number of facial keypoints = %d' %(numKeypoints))
# number of remaining images = 2140
# image dimentions = (96,96)
# number of facial keypoints = 15

# Some preprocessing 
Y = facialKeypointsDF.as_matrix()
Y_mean = Y.mean()
Y_std = Y.std()
Y = (Y - Y_mean) / Y_std

X = np.transpose(faceImagesDB, (2,0,1))
X/=255
X = X.reshape((-1,96,96,1))
```

Now we are ready for the training. We will create a simple CNN for this task. 

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(96,96,1)))
model.add(Conv2D(32,kernel_size=(3,3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='linear'))
sgd = SGD(lr=0.0001,momentum = 0.9,nesterov=True)
model.compile(loss="mean_squared_error",optimizer=sgd)
model.fit(X,Y,epochs=20,validation_split=0.2,verbose=1)
```

So after the training we can simple predict the landmarks and use it in the future. Good things about this method is we dont need to train it for the new people. 

![](../images/post5/2.png)

Blue dots are predicted values and red ones are grount truth. As we can see even simple network is quite good at marking them. And here is a predicted values for unseed pictures. 

Although it is not the best way to implement this, this method is also works just fine. Idealy We should use triple loss fucntion to create embdeings. Triple loss is an Idea of finding best parameters to seperete and recognize people. In nutshell, We want network to get same persons embedings closer and if they are diffrent then seperate them in multidimentional space. For implementation we need three photos. Two from same person and One from someone else. We will just fit the network and after the forward pass, we will check how close same persons face embedings are and How far with the diffrent person. So with this loss network will be learning to separate people. 



Okay, since we solved the face recognition, we have another problem. What happens if someone holds a picture of me to get in. or How can we make sure they are live people and not someone with a mask. So we just need to classify if the input frames contain live people. For this stage, we can use a 3D convolutional network in our model. Why 3D you asked? Cause a frame does not contain some very useful features like eye movement, winkings etc. RNN also would work great but that part on you if you want to try out. (Please don't forget to send it to me 😄) 

Simple implementation of the liveness detection is as follows: 

```python

# X contains [n, 24, 100, 100, 1] arrays of image. Which is 24 frame long video in gray scale
# Y containt [n, 2] labels for per video input in one hot 

# Model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3),
                 activation='relu',
                 input_shape=(24,100,100,1)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


```

Next step is basic supervised training. 

When you are using in the sytem you also need to input a 24 frames long array. So first you need to skip 23 frames and save it in the memory then your can start predicting the liveness of the input. 



Now, these two big problems should be solved. Please note that these two methods are not the best but they get the job done, but just keep in mind that I wouldn't use this project on my houses entrance 😅  Just use it in the meeting room or something like office(if you dont keep something valuable. 😄 


In conculution, You can find all this code in [this repository](https://github.com/AhmetHamzaEmra/Intelegent_Lock).

Have fun! 

### TLDR:

- AI & ML good
- Landmark model for face recognition is crucial 
- Liveness is also crucial
- Not the safest lock  (at least for now)