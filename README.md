# Reinforement-Learning-on-Taxi-Problem
<p> 
  This project explores the implementation of SARSA and Q-learning algorithms on the Taxi problem from the OpenAI Gym Toy_text environment. The aim is to train optimal policies for the taxi agent and analyze the computational efficiency and reward maximization capabilities of the two algorithms.
</p>

## Description: 
<p>
  In this environment, there are four designated locations represented by R(ed), G(reen), Y(ellow), and B(lue). At the beginning of each episode, the taxi and the passenger are randomly positioned in the grid world. The taxi's task is to navigate to the passenger's location, pick up the passenger, drive to the passenger's destination, and finally drop off the passenger. The episode ends when the passenger is successfully dropped off.
</p>

### Actions:
<p>
  The action space consists of 6 discrete deterministic actions:
  </p>

0: Move south <br />
1: Move north <br />
2: Move east  <br />
3: Move west <br />
4: Pickup passenger <br />
5: Drop off passenger

### Observations:
<p>
The observation space consists of 500 discrete states. Each state is represented by a tuple: (taxi_row, taxi_col, passenger_location, destination). There are 25 possible taxi positions, 5 passenger locations (including when the passenger is in the taxi), and 4 destination locations. However, there are 400 states that can be reached during an episode, as some states correspond to situations where the passenger is already at their destination. Additionally, four additional states can be observed when both the passenger and the taxi are at the destination, right after a successful episode. </p>

### Rewards:
<p>
  The project implements SARSA and Q-learning algorithms to train optimal policies for the Taxi problem. SARSA is an on-policy algorithm that updates the Q-values based on the current policy, while Q-learning is an off-policy algorithm that updates the Q-values using the maximum Q-value of the next state. By comparing the two algorithms, we analyze their computational efficiency in terms of convergence time and their ability to maximize the cumulative reward. </p>

## Implementation:
<p> The project implements SARSA and Q-learning algorithms to train optimal policies for the Taxi problem. SARSA is an on-policy algorithm that updates the Q-values based on the current policy, while Q-learning is an off-policy algorithm that updates the Q-values using the maximum Q-value of the next state. By comparing the two algorithms, we analyze their computational efficiency in terms of convergence time and their ability to maximize the cumulative reward. <p/>

## Project Structure:
<li> `QLearning.py`: Implements the Q-learning algorithm for training the optimal policy. </li>
<li> `SarsaLearning.py`: Implements the SARSA algorithm for training the optimal policy. </li>
<li> `plot.py`: contains code to generate plots for the comparative analysis of the Q-learning and SARSA learning algorithms, providing insights into their performance on the Taxi problem. </li>

## Comparative Analysis: 
<p> A comparative analysis of the SARSA and Q-learning algorithms was conducted to evaluate their performance on the Taxi problem. The analysis focused on two main aspects: </p>

1. **Computational Efficiency:** The convergence time of the two algorithms was measured to assess their computational efficiency. Convergence time refers to the number of episodes required for the algorithms to converge to an optimal policy. The algorithm that achieved convergence in fewer episodes was considered more computationally efficient.<br/>

2. **Maximizing Reward:** The learned policies from both SARSA and Q-learning were executed from various starting states, and the average reward was calculated for each algorithm. The average reward represents the effectiveness of the learned policy in maximizing the cumulative reward over multiple episodes. The algorithm with a higher average reward was considered better at maximizing rewards.
<p>The results of the comparative analysis can be found in the attached report. The analysis includes graphs, tables, and discussions highlighting the performance of SARSA and Q-learning in terms of computational efficiency and reward maximization. <p/>
