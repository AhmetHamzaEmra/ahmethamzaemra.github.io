# Ensemble Modeling 



## What is it? 

It is basicaly combining many models to get better predictions. There are diffent methods, we can use them. Such as; 

* Averaging (or blending)
* Weighted averaging 
* Conditional averaging 
* Bagging 
* Boosting 
* Stacking 
* etc.

Lets get stared with averaging. 

### Averaging 

![post4.1]()

Lets say we have this data. As we can see our predicitons are quite well under age 50 but after that point we tent to miss the predicitons. Lets say we have another model which looks like this; 

![post4.2]()

it is oblivious that this models oposite power to each other. so what we do in this case is we can take the average. 

```python
(model1_predicitons + model2_predictions ) / 2 
```



