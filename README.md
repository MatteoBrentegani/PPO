# Proximal Policy Optimization - with Keras

This is Keras implementation of PPO:
  * double action stream (Linear and Angular Velocity) [PPO](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_DoubleAction)
  * multiple agent [PPO-MultipleAgent](https://arxiv.org/pdf/1707.06347.pdf)



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

 * Python 3
 * Tensorflow
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

For run the code just run the main.c file into anaconda environment:

```
activate tensorflow
python main.c
```
(If you change environment you should change the imported environment into main.c)

End with an example of getting some data out of the system or using it for a little demo

