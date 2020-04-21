# PPO with centralized neural network

# Summary
This code implement a version of PPO with multiple agent and a shared neural network.

# Code
* ``main.py``

Create environmentt and run the ppo algorithm.

* ``config.yaml``

Configuration about agent and environment

* ``ppo_lossFuncitionC.py``

Implementation of PPO algorithm.


### performance examples


#### PPO
![image](https://github.com/MatteoBrentegani/PPO/blob/master/PPO_CentralizedNN/results/PPO_centralized.png)

In the figure we can see the behavior of the network when an agent manages to reach the goal.
When this behavior does not occur the agents remain stationary in the starting position.

Furthermore, by increasing the number of agents, this behavior occurs constantly.
This is due to the fact that the network outputs require a weight.

* ``ppo_lossFuncitionC.py``

``
lossWeights={
    "output11":0.3125,
    "output12":0.1875,
    "output21":0.3125,
    "output22":0.1875
}
``

These are necessary when the network has multiple outputs. The values ​​used were obtained from the tests carried out on the [PPO with double actions](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_DoubleAction).




