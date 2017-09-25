---
title:  "Machine Learning Yearning!"
date:   2017-09-24 15:04:23
published: true
categories: [Machine Learning]
tags: [ML, Machine Learning, AI, Deep Learning]
---
By Andrew Ng

![img](https://images.gr-assets.com/books/1480798569l/30741739.jpg)


I have recently find out the handbook, from Dr. Andrew Ng. He shared the draft version of his book. The goal of this book is to teach you how to make the numerous decisions needed with organizing a machine learning project. In this post, I would like to share some of my findings.

### 1. Many of the ideas of deep learning (neural networks) have been around for decades. Why are these ideas taking off now?

Two of the biggest drivers of recent progress have been:
- Data availability. People are now spending more time on digital devices (laptops, mobile devices). Their digital activities generate huge amounts of data that we can feed to our learning algorithms.
- Computational scale. We started just a few years ago to be able to train neural networks that are big enough to take advantage of the huge datasets we now have.


<img src="/images/post3/1.png">


### 2. Your development and test sets

We usually define:
- Training set — Which you run your learning algorithm on.
- Dev (development) set — Which you use to tune parameters, select features, and make other decisions regarding the learning algorithm. Sometimes also called the hold- out cross validation set.
- Test set — which you use to evaluate the performance of the algorithm, but not to make any decisions about regarding what learning algorithm or parameters to use.

Choose dev and test sets to reflect data you expect to get in the future and want to do well on.


### 3. Your dev and test sets should come from the same distribution

Once you define the dev and test sets, your team will be focused on improving dev set performance. Thus, the dev set should reflect the task you want most to improve on: To do well on all four geographies, and not only two.

As an example, suppose your team develops a system that works well on the dev set but not the test set. If your dev and test sets had come from the same distribution, then you would have a very clear diagnosis of what went wrong: You have overfit the dev set. The obvious cure is to get more dev set data.

### 4. How large do the dev/test sets need to be?

Dev sets with sizes from 1,000 to 10,000 examples are common.

### 5. Establish a single-number evaluation metric for your team to optimize & Optimizing and satisficing metrics

Having a single-number evaluation metric speeds up your ability to make a decision when you are selecting among a large number of classifiers. It gives a clear preference ranking among all of them, and therefore a clear direction for progress.
 
Optimizing metric = Goal metric which we care about most 

Satisficing metric = Simply require that they meet a certain value 

### 6. Having a dev set and metric speeds up iterations

1. Start off with some idea on how to build the system.
2. Implement the idea in code.
3. Carry out an experiment which tells me how well the idea worked. (Usually my first few ideas don’t work!) Based on these learnings, go back to generate more ideas, and keep on iterating.


<img src="/images/post3/2.png">

### 7. When to change dev/test sets and metrics

If you later realize that your initial dev/test set or metric missed the mark, by all means change them quickly. For example, if your dev set + metric ranks classifier A above classifier B, but your team thinks that classifier B is actually superior for your product, then this might be a sign that you need to change your dev/test sets or your evaluation metric.

There are three main possible causes of the dev set/metric incorrectly rating classifier A higher:

1. The actual distribution you need to do well on is different from the dev/test sets.
2. You have overfit to the dev set.
3. The metric is measuring something other than what the project needs to optimize.


Thank you & See you later! 



