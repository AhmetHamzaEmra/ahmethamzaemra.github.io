---
layout: post
title: "Ensemble Modeling for beginers"
tags: [Machine Learning]
<!-- feature-img: "assets/img/sample_feature_img.png" -->
---


It is combining many models to get better predictions. There are diffent methods, we can use them. Such as; 

* Averaging (or blending)
* Weighted averaging 
* Conditional averaging 
* Bagging 
* Boosting 
* Stacking 
* etc.

Lets get stared with averaging. 

### Averaging 

Lets say we have a data and our predicitons are quite well under age 50 but after that point we tent to miss the predicitons. And we have another model which works better after 50 and not good between 0 to 50. We can basicaly take the average of out predictions.  

```python
predictions = (model1_predicitons + model2_predictions ) / 2 
```

### Weighted & Conditional averaging 

``` python
predictions = (model1_predicitons * Weight1 + model2_predictions * Weight2 ) / 2 
```

Sometimes multiblying the predicitons with some weight and taking average helps but almost every time we need to understand the models and data completely. Otherwise this methods are hard to get benefit from. Another way to create ensemble model is conditional averaging. What we do in this one is take the strong part of models and make preditctions according to conditions.

``` python
if age < 50:
	predictions = model1_prediction
else:
	predictions = model2_prediction
```

### Bagging

Bagging means averaging slightly diffent version of the same model to improve accuracy. For example, Random forrest which runs diffent decison trees and combine the scores. Basicaly we can create bagging with:

* Changing the seed
* Sub-sampling rows 
* Sub-sampling column 
* Changing model-specific parameters
* Number of models 

Here is example code where we change the seed:

```python
# train is the training data
# test is the test data
# y is the target variable 
model = RandomForestRegressor() # from sklearn 
bags = 10
seed = 1
# create a array object to hold bagged predictions 
bagged_prediction = np.zeros(test.shape[0])
# lood through bags 
for n in range(0, bags):
  model.set_params(random_state=n+seed) # update seed
  model.fit(train, y) # fit the model 
  preds = model.predict(test) # predict on the test cases
  bagged_prediction += preds # add predictions to the bagged predictions 
# take the average of the predictions 
bagged_prediction /= bags  
```

### Boosting 

A form of weighted averaging of models where each model is built sequentially via taking into account the past model performance. It takes considiration to previous models performace. Lets go over it with an example. 

#### Weight based boosting 

in this type boosting, what generaly done is we fit one model and get predict probabilities. After that we can find the abs. error and can create a weight column using errors. Now we can see on which columns our model done worst and try to minimize the error on the next model. So that we can create a better model sequentially. there are some parameters which we need to optimize. 

* Learning rate (we should not trust every predictions right away. reduce the trust with learning rate:  ` predictionN = learning_rate*pred0 + learning_rate*pred1 + …) `
* Num of extimators 
* Input model - can be anything that accept weights 
* Sub boosting type: 
  * AdaBoost - Good implementation on sklearn
  * LogitBoost

#### Residual based boosting 

In this type, we find the arror but not absolute. Than we make this error column to new target variable. the predictions coming from second model are being added to our first predictions. What we endup with is a predictions that are closer to initial target values. Parameters:

* Learning rate (we should not trust every predictions right away. reduce the trust with learning rate:  ` predictionN = pred0 + learning_rate*pred1 + learning_rate*pred2 + …) `
* Num of extimators 
* Sub-sampling rows 
* Sub-sampling column 
* Input model - better be trees but can be anything 
* Sub boosting type: 
  * Fully gradient based 
  * Dart (very efficient with classification)

**Some good implementations of Residual based boosting:**

* Xgboost
* Lightgbm
* H2O's GBM
* Catboost
* Sklearn's GBM

### Stacking 

Means making predictions of a number of models in a hold-out set and then using diffent (meta) model to train on these predistions. 

Methodology 

1. Splitting the train set in to two disjoint sets.
2. Train several (base) learners on the first part. 
3. Make predicsitons with base learniers on the second (validation) part.
4. Using the predictions from (3) as the input and train a higher level learner.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np 
from sklearn.model_selection import train_test_split

# train is the training data
# y is the target variable for training data
# test is test data

# split train data in 2 parts, training, and validation
training, valid, ytraining, yvalid = train_test_split(train, y, test_size=.5)
# specify models 
model1 = RandomForestRegressor()
model2 = LinearRegression()
# fit models
model1.fit(training,ytraining)
model2.fit(training,ytraining)
# make predictions for validation
preds1 = model1.predict(valid)
preds2 = model2.predict(valid)
# make predictions for test data
test_preds1 = model1.predict(valid)
test_preds2 = model2.predict(valid)
# form a new dataset for valid and test viastacingthe predictions 
stacked_predictions = np.column_stack((preds1,preds2))
stacked_test_predictions = np.column_stack((test_preds1,test_preds2))
# specify meta model 
meta_model = LinearRegression()
# fit meta model on stacked predictions
meta_model.fit(stacked_predictions, yvalid)
# make predictions on the stacked predictions of the test data 
final_predictions = meta_model.predict(stacked_test_predictions)

```



Things to be mindful of:

* with time sensitive data - respect time
* Diversity as important as performance
* Diversity may come from:
  * Different algorithms
  * Different input features 
* Performance pleteauing after N models
* Meta model should be simple 



## This all I want to share in this post, Untill next time :D





