# Proximal Policy Optimization - with Keras

This is an implementation of PPO with:
  * double action stream (Linear and Angular Velocity) [PPO_DoubleAction](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_DoubleAction)
  * multiple agent [PPO-MultipleAgent](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_MultiAgent)



## Summary

The project implement the clipped version of Proximal Policy Optimization Algorithms described here https://arxiv.org/pdf/1707.06347.pdf

Into the config.yaml file are defined some of the hyper parameter used into the various implementation. Those parameters are initilized with the values proposed in the article.


Here the key point:
* Loss function parameters:
  * epsilon = 0.2
  * gamma = 0.99
  * entropy loss = 1e-3
  
* Network size:
  * state_size = 27 (25 laser scan + target heading + target distance)
  * action_size (angular velocity) = 5
  * action_size2 (linear velocity) = 3
  * batch_size = 64
  * output layer = 8 into 2 streams (5 nodes for angular and 3 for linear velocity)
  * lossWeights for the output layer: 
    * 0,625
    * 0.375
    
The values for the loss weights are the result of some test. With an equal weight the success rate is lower. 

### Prerequisites

 * Python 3
 * Tensorflow
 * NumPy, matplotlib, scipy
 * [Keras](https://keras.io/)
 * [Unity](https://unity3d.com/get-unity/download)

```
# create conda environment named "tensorflow"
conda create -n tensorflow pip python=3.6

# activate conda environment
activate tensorflow

# Tensorflow
pip install tensorflow

# Keras
pip install keras
```

### Training

For start the training run the main.py file into anaconda environment:

```
activate tensorflow
python main.c
```




