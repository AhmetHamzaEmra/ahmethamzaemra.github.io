---
layout: post
title: "What happened next?"
tags: [Deep Learning, Artificial General Intelengece]
---



Movies are one of those ways which take people anywhere they want to be. Sometimes it brings you an unexplainable smile on your face and sometimes you feel the sadness of the character you loved.  But not always movies explain the ending. It gives some space to watchers to starch the imagination. It is always fun to hear other peoples fan theories and argues about it. I wonder what would machines think...



With the rise of deep learning, humanity got some of the satisfying answers and we are still having surprises. That being said, In this project, I try to get an answer to "What happened next?" 

## How it works 

Before I share the results, I would like to add some info about what is going on behind the scene. Holding knowledge and understanding concepts is very easy for us humans to perform. But this tasks are extremely challenging for machines to compleate. Even though this difficulties, with the [Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35), Software engineers and researchers solved many problems on the way to reach the state they dreamed. One of this dreams is generating a text. With many improvements in the area like LSTM network and Word Embedings, Researchers started to get good results in text generation. In this article I will not go in to technical details too much. If you want to learn how this networks work please read this [post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.  But what it does in short is, it takes the words and trasnform them into embeding vectors. Then with the Recurrent Neural Network we predict the next work. The basic structure is as it shown below:

![](../images/whnn.png)

## Results



Now I would like to share some of the scripts that are generated. Let's go to and check a couple of movies and see the answers:

### Inception

Every body has an opinion about inception. Movie has everything, mindblowing script and unexpected ending. But What I wonder is what does AI thinks is going to happen next. And here is the generated script after training the network with script of the movie:

> knocks on the glassjames and philippa turn- see their dad . he steps to the window , watching their bright faces shining as they run towards him ... behind him , on the table , the spinning top is still spinning . and wefade out . credits . end . **mal is pointing at the ceiling . mal why do you do it ? mal thinks , unable to remember . a faint rumble begins . ariadne oh my god . *we 're dreaming* .**

Well it seems clear that AI thinks they are in a dreem. 

### (500) Days of Summer

 The question is; After everything that happened to Tom with Summer, did he find what he was looking for with Autumn? 

Here is a couple of lines that our network generated:

> *he(Tom) 's ripping at posters and Edward Hopper*
>
> *tom is powerless to stop a frustration from being audible*
>
> *summer pathetic*

Unfortunately, it seems Tom is not happy after all :cry:



## So what else?



I wrote a 120 lines of python code, which can be found in this [here](https://github.com/AhmetHamzaEmra/What_happened_next). Feel free the experiment and please share the cool results with me :smile: Honestly, this work needs a lot of improvement, so feel free to contribute...