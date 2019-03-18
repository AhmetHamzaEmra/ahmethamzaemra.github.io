---
layout: post
title: "Reinforcement learning with Tensorflow 2.0"
tags: [Deep Reinforcement Learning, Artificial Intelengece]
---




![](https://cdn-images-1.medium.com/max/1600/1*-GpYqxeC6lef8WNaOhEidw.png)

Reinforcement learning is a fascinating field in artificial intelligence which is really on the edge of cracking real intelligence. With the new Tensorflow update it is more clear than ever. In this series, I will try to share the most minimal and clear implementation of deep reinforcement learning algorithms.

Please check out [this post](https://medium.com/tensorflow/test-drive-tensorflow-2-0-alpha-b6dd1e522b01) to get started with Tensorflow 2.0

Before we begin to code, let’s start with what is Reinforcement learning and what makes it different. In most general terms RL is What you do — How to map situations to actions — so as to maximize the numerical reward signals or minimize penalties. It is basically encouraging action with good reward and discouraging actions with any penalties. It is the same as how we, human, and animals learn.

Reinforcement learning is part of machine learning. Although it has some similarities with both supervised and unsupervised learning, it is a third separate field. What makes it different is not needing an external supervisor and the concept of a goal.

### Policy Gradient

Today we will go over one of the widely used RL algorithm Policy Gradients. this method is using a neural network to complete the RL task. Let’s go over step by step to see how it works.

The neural network is one of the best practice to use in supervised learning. It can map features to labels while extracting useful new features on the way. But the challenge with reinforcement learning is we don’t know what is good output. So the agent (player) needs to figure it out by itself. Even though this is a big deal, we have something that can give us clues on determining which actions are good. which is a reward function. Now, our goal is to make actions with good reward more encouraged and actions with negative reward should be discouraged. In most vanilla approach, we can do this by adjusting our gradient according to the reward we get. If you want to learn more about policy gradient methods please check out [this video.](https://www.youtube.com/watch?v=S_gwYj1Q-44)

### Credit Assignment Problem

One last thing before we start implementations, let’s go over the concept of credit assignment problem. The issue in here is we don’t know which action resulted in that reward. It might be because we did a really good couple of seconds ago and the reward is just delayed. There is no way of knowing which action is responsible for what but we can try to give all of them some part. So what we need to do is go back in all actions and give some degree of reward for each one. This process is called discounting rewards. Here is a small script that does it for us:

```python
# gamma: discount rate 
def discount_rewards(r, gamma = 0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
```

### Implementation with Tensorflow 2.0

With the new functions and eager execution, it is easier than ever before. Let’s start with building our network. In here we just initialize NN with one hidden layer, optimization algorithm and a loss function.

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim = 4, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

Before we start training we need another variable to hold our gradients when we do the backpropagation. Later this will be use-full when we want to make changes in gradients

```python
gradBuffer = model.trainable_variables
for ix,grad in enumerate(gradBuffer):
  gradBuffer[ix] = grad * 0
```

Now our training loop:

```python
for e in range(n_episodes):
  # reset the enviroment
  s = env.reset()
  ep_memory = []
  ep_score = 0
  done = False 
  while not done: 
s = s.reshape([1,4])
    with tf.GradientTape() as tape:
      #forward pass
      logits = model(s)
      a_dist = logits.numpy()
      # Choose random action with p = action dist
      a = np.random.choice(a_dist[0],p=a_dist[0])
      a = np.argmax(a_dist == a)
      loss = compute_loss([a], logits)
    grads = tape.gradient(loss, model.trainable_variables)
    # make the choosen action 
    s, r, done, _ = env.step(a)
    ep_score +=r
    if done: r-=10 # small trick to make training faster
    
    ep_memory.append([grads,r])
  scores.append(ep_score)
  # Discound the rewards 
  ep_memory = np.array(ep_memory)
  ep_memory[:,1] = discount_rewards(ep_memory[:,1])
  
  for grads, r in ep_memory:
    for ix,grad in enumerate(grads):
      gradBuffer[ix] += grad * r
  
  if e % update_every == 0:
    optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0
      
  if e % 100 == 0:
    print("Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))
```

Lets explain the code little bit:

```python
s = s.reshape([1,4])
with tf.GradientTape() as tape:
  logits = model(s)
  a_dist = logits.numpy()
  # Choose random action with p = action dist
  a = np.random.choice(a_dist[0],p=a_dist[0])
  a = np.argmax(a_dist == a)
  loss = compute_loss([a], logits)
grads = tape.gradient(loss, model.trainable_variables)
```

This part is forward propagation and action selection part. The reason we do random action selection is an exploration of the environment. If we choose the best action every time, we cannot be sure to know that is the optimal solution. Also, please note that we calculate the gradients but not apply them.

Next part is applying the gradient:

```python
# Discound the rewards 
  ep_memory = np.array(ep_memory)
  ep_memory[:,1] = discount_rewards(ep_memory[:,1])
  
  for grads, r in ep_memory:
    for ix,grad in enumerate(grads):
      gradBuffer[ix] += grad * r
  
  if e % update_every == 0:
    optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0
```

Before we apply the gradient we need to calculate the advantage score. Advantage score is discounted reward for this case. Ergo, the discounted score is calculated then we multiply those scores with our gradient. After a couple of episodes of collecting gradient, we apply and change the weights of our network.

Lastly, we check the average 100 episodes mean score:

![](https://cdn-images-1.medium.com/max/1600/1*43v5bIKKIsORik0CMgI3VA.png)



![](https://cdn-images-1.medium.com/max/1600/1*p_S90VBL5k6L0dHPqzrd4Q.gif)

So this is it, folks! It is really simple and you can apply this to any problem you want. You can find the code in the link below. Until next time!

[Colab notebook](https://colab.research.google.com/drive/1OBZMWLSU-jZuXWCps16EDNbxnv4Gl-wr)

