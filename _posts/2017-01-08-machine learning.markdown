---
title:  "Machine Learning for Beginners!"
date:   2017-03-12 15:04:23
published: true
categories: [Machine Learning]
tags: [ML, Machine Learning, AI, Regression]
---


Since Artificial Intelligence is a hot topic right now, Machine Learning’s popularity is increasing. In this post, I will be summarizing what is machine learning and where to start.


## What is machine learning?


Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.

For example, the you have data set like this,


| Bathroom        | Sq. Feet    | Price  |
| :-------------: |:-------------:| -----:|
| 1     |800     |  $1600 |
| 2     | 1000   |  $1500 |
| 5     | 4000   |    $10000 |
| 3     | 800    |    $1200 |
| 4     | 4000   |    $8000 |



and you want to find out house that has 6 bathrooms and 7000, we don’t need to define extra functions or find out relation or formula of price. We just push data to our Machine Learning algorithm and it figures out for us.


``` python
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures


X = [[1,800],[2,1000],[5,4000],[3,800],[4,4000]] # What we are teaching with (#Bathroom, sq.feet)
Y = [1000,1500,10000,1200,8000] # What we want to find out 


poly = PolynomialFeatures(degree=2) # converting dat ato polynomial 
X = poly.fit_transform(X)

reg = LinearRegression() # Main algorithm 
reg.fit(X,Y) # Theaching with our examples

test_case = poly.fit_transform([6,7000]) # converting test case to same format as our training cases
prediction = clf.predict(test_case) # Make a predictions 
print(prediction) # result = $18389,6

```

So the house we are looking for is $ 18389,6. It will be very hard to calculate it mathematically. (so , I will not do it :D )

This is one of the algorithm called Regression. Documentation can be found in this [link](http://scikit-learn.org/stable/modules/linear_model.html ). I will go over other machine learning types in next posts.

Regression problems are great point to start. They are relatively easy and has a lot of practice opportunity. Especially, Hackerrank.com has great AI section that contains very nice machine learning questions. There are bunch of questions for every level.

I will try to upload more post soon. Please share your knowledge, ask questions, make contributions. See you soon…