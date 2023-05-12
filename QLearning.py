import numpy as np
import gym
import random
import copy

# If want to see demo of taxi, Put RENDER = 1
RENDER = 1   
def QLearning_train(epsilon,learning_rate,discount_rate,ep):
    # create Taxi environment
    env_train = gym.make("Taxi-v3")

    state_size = env_train.observation_space.n  # 500 state
    action_size = env_train.action_space.n      # 6 actions
    
    # initialize q-table 
    qtable = np.zeros((state_size, action_size))
    
    # training variables
    num_episodes = ep
    max_steps = 99  # per episode
    episode = 1
    
    # training
    for i in range(num_episodes):
        # reset the environment
        state, _ = env_train.reset()
        done = False
        q = copy.deepcopy(qtable)
        for s in range(max_steps):  
            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:                      # Epsilon Greedy
                # explore
                action = env_train.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :]) # this cannot be same as greedy policy!

            # take action and observe reward
            new_state, reward, done, info, _ = env_train.step(action)
            

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            
                      
            # Update to our new state
            state = new_state

            # if done, finish episode
            if done==True:
                break

        episode+=1
    
    env_train.close()
    
    print(f"Training completed over {episode} episodes")
    return qtable
    
def QLearning_test(qtable,no_of_demo):
    if RENDER == 1:
        env_test = gym.make("Taxi-v3",render_mode='human')
    else: 
        env_test = gym.make("Taxi-v3")
    
    max_steps = 99
    
    score_lst = []
    steps_lst = []
    
    for i in range(no_of_demo):
        state, _ = env_test.reset()
        done = False
        rewards = 0

        for s in range(max_steps):
            # print(f"TRAINED AGENT")
            # print("Step {}".format(s + 1))

            action = np.argmax(qtable[state, :])
            new_state, reward, done, info, _ = env_test.step(action)
            rewards += reward
            
            # env_test.render()
            # print(f"score: {rewards}")
            state = new_state

            if done == True:
                break
        # print("Total steps taken : ", s)  
        # print(f"Total score: {rewards}") 
 
        score_lst.append(rewards)
        steps_lst.append(s)
    
 
    env_test.close()
    
    return score_lst, steps_lst

def QLearning(epsilon,learning_rate,discount_rate,ep,no_of_demo):
    Q_table = QLearning_train(epsilon,learning_rate,discount_rate,ep)
    rewards, steps = QLearning_test(Q_table,no_of_demo)
    avg_reward = np.mean(rewards) 
    avg_steps = np.mean(steps)
    std_reward = np.std(rewards)
    std_steps = np.std(steps)
    
    return avg_reward,std_reward,avg_steps,std_steps

if __name__ == "__main__":
    QLearning(epsilon=0.5,learning_rate=0.9,discount_rate=0.9,ep=1000,no_of_demo=10)