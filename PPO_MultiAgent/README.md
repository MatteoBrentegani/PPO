# Proximal Policy Optimization with multiple agent

This project implement 7 agent and a single neural network. The algorithm work as always for each agents, 
but the neural network will be trained with only one agent experience. 
This is an agent that reached a final state (collision/goal). The goal is only achieved if an agent reaches his cube.

<img src="https://github.com/MatteoBrentegani/PPO/blob/master/PPO_MultiAgent/MA_base.gif" width="250" height="250"/>

# Code
* ``main.py``

Create environmentt and run the ppo algorithm.

* ``config.yaml``

Configuration about agent and environment

* ``ppo_lossFuncition.py``

Implementation of PPO algorithm.


# Training

For the training were used two environment. The first one without obstacle and the second with 4 walls.


### performance examples
#### PPO Multiple Agents
![image](https://github.com/MatteoBrentegani/PPO/blob/master/PPO_MultiAgent/results/ppo_MultiAgent.png)


#### PPO Multiple Agents with obstacle
![image](https://github.com/MatteoBrentegani/PPO/blob/master/PPO_MultiAgent/results/ppo_MultiAgentObstacle.png)

It easy to notice that obstacles decreased the success rate. In any case, the behavior of the algorithm applied to a [single agent](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_DoubleAction) does not differ much from that obtained here.


