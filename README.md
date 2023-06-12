# Reinforement-Learning-on-Taxi-Problem
<p> 
  This project explores the implementation of SARSA and Q-learning algorithms on the Taxi problem from the OpenAI Gym Toy_text environment. The aim is to train optimal policies for the taxi agent and analyze the computational efficiency and reward maximization capabilities of the two algorithms.
</p>
## Description: 
<p>
  In this environment, there are four designated locations represented by R(ed), G(reen), Y(ellow), and B(lue). At the beginning of each episode, the taxi and the passenger are randomly positioned in the grid world. The taxi's task is to navigate to the passenger's location, pick up the passenger, drive to the passenger's destination, and finally drop off the passenger. The episode ends when the passenger is successfully dropped off.
</p>

## Actions:
<p>
  The action space consists of 6 discrete deterministic actions:
  </p>
  
Reinforcement Learning Project: Taxi Problem
This project explores the implementation of SARSA and Q-learning algorithms on the Taxi problem from the OpenAI Gym Toy_text environment. The aim is to train optimal policies for the taxi agent and analyze the computational efficiency and reward maximization capabilities of the two algorithms.

Description
The Taxi problem is adapted from the paper "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition" by Tom Dietterich. In this environment, there are four designated locations represented by R(ed), G(reen), Y(ellow), and B(lue). At the beginning of each episode, the taxi and the passenger are randomly positioned in the grid world. The taxi's task is to navigate to the passenger's location, pick up the passenger, drive to the passenger's destination, and finally drop off the passenger. The episode ends when the passenger is successfully dropped off.

Actions
The action space consists of 6 discrete deterministic actions:

0: Move south
1: Move north
2: Move east
3: Move west
4: Pickup passenger
5: Drop off passenger
