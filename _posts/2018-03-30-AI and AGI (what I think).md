---
layout: post
title: "AI and AGI with Deep learning"
tags: [Deep Learning, Artificial General Intelengece]
---

I sometimes see people view me as deep learning obsess and I just want to share some beliefs of mine in context of deep learning. 

I would like to start with mentioning the good parts of deep learning then we can talk about the missing parts. But before I start I must say that I am not domain expert (yet 😄), so please take it with a grain of salt. 

## What to love?

### Representation Learning!

Machine learning (supervised learning) is basically finding the mapping between the representations of data to labels. And there are many algorithms which can produce this function estimation but one of the biggest performance dependency is features. Good features mean good performance! But should we create hand designed features or should model discovers them? To be able to answer this question, first, we need to set our goals.

My personal view and expectancy from AI is that it should complete the given task with as little as possible (or no) human interaction. So let's get back to the question, model should discover the features from data. Ergo, DL is the best performing method in both supervised and unsupervised learning tasks. Also, I believe, Ability to learn useful information is a big step towards to AGI.


### Transfer Learning!  

As I mentioned in the last paragraph, representation learning is a huge advantage of deep learning among other methods. But another thing that makes human learning much faster and efficient is the ability to transfer knowledge into other fields. Let's think about it, we trained a very large network with a large dataset for 2-3 weeks and at the end, it has no help for subtask or similar problem. We will end up with training new network for every task. I am oblivious that this is nowhere human-like learning and very inefficient. On the other hand, this is not a case for deep learning. Neural networks allow us to transfer some its knowledge into other problems. If you are curious about how it is done, please visit [cs231n transfer learning lecture by Justin Johnson.](https://youtu.be/_JB0AO7QxSA?t=1h9m37s)



For now, this two factor should be enough to show how awesome Deep learning is! 

## On the other hand...

Lets touch couple problem with deep learning;

### Not enough data? BIG PROBLEM!

Most of the current popular algorithm in Neural Network requires a lot of data. Which is odd because if we are trying to learn like a human, it should not require that much! I am aware of research progress in methods like zero-shot learning, one-shot learning etc. but what everybody (including me) use today is not the optimal form of learning! So I believe this is one of the main flaws of deep learning.

On another point of view, we are able to correct mistakes with reinforcement learning but actually, the finding the points where we completely messed up and being able to fix it, like in Boosting methods, is another thing humans are outperforming DL models.


### Only Pattern recognition?

Deep learning created a revolution in the field of AI but how about AGI. As [Josh Tenenbaum said in his talk at MIT](https://www.youtube.com/watch?v=7ROelYvo8f0), all those great machines are just pattern-recognition systems. For example, AlphaGo is the best in its domain, but it can not even say "What GO is?" So having a general concept of the World is another problem, we as an AI community need to solve. Another great example is from [Andrej Karpathy in his blog post](https://karpathy.github.io/2012/10/22/state-of-computer-vision/), He pointed how these methods are not capable of understanding all concept rather than making a gist of sense. 

### Closing thoughts

Overall, Deep Learning definitely has potential to solve this issues. Many great advancements have been made in the area which proves potential. For instance, DNC, World Models, and many others. In the end, This is what I believe in a nutshell, we are going to see how it is going to end up.
