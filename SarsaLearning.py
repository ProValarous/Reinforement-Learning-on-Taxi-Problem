import numpy as np
import gym
import random
import copy

# def Select_action(env_train, qtable, epsilon, state):
#     action = np.argmax(qtable[state, :]) # greedy policy!

#     return action

def SarsaLearning_train(learning_rate,discount_rate,ep):
    # create Taxi environment
    env_train = gym.make("Taxi-v3")

    # initialize q-table
    state_size = env_train.observation_space.n           # 500 states
    action_size = env_train.action_space.n               # 6 actions
    qtable = np.zeros((state_size, action_size))
    
    # training variables
    num_episodes = ep
    max_steps = 99  # per episode
    episode = 1
    
    # convergence_threshold_count = 0
    
    # training
    for i in range(num_episodes):
        # reset the environment
        state, _ = env_train.reset()
        done = False
        q = copy.deepcopy(qtable)
        for s in range(max_steps):  
            # exploration-exploitation tradeoff
            action = np.argmax(qtable[state, :])                # greedy policy!
            # action = env_train.action_space.sample()
            # take action and observe reward
            new_state, reward, done, info, _ = env_train.step(action)
            next_action = np.argmax(qtable[new_state, :])       # greedy policy!
            # Sarsa 
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * qtable[new_state,next_action] -qtable[state,action])
            # Update to our new state
            state = new_state
            action = next_action
            # if done, finish episode
            if done==True:
                break
        
        # if np.array_equal(qtable,q):
        #     convergence_threshold_count+=1  
        # else:
        #     convergence_threshold_count=0        
        
        # if convergence_threshold_count > 3:
        #     break
        
        # Decrease epsilon
        # epsilon = np.exp(-decay_rate * episode)
        episode+=1
    
    env_train.close()

    print(f"Training completed over {episode} episodes")
    return qtable

def SarsaLearning_test(qtable,no_of_demo):
    # env_test = gym.make("Taxi-v3",render_mode='ansi')
    env_test = gym.make("Taxi-v3")

    max_steps = 99

    score_lst = []
    steps_lst = []

    for i in range(no_of_demo):
        state, _ = env_test.reset()
        done = False
        rewards = 0
        # print("Demo # ",i+1)
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
        # print("#################")   
        score_lst.append(rewards)
        steps_lst.append(s)

    env_test.close()
    return score_lst, steps_lst

def SarsaLearning(learning_rate,discount_rate,ep,no_of_demo):
    Q_table = SarsaLearning_train(learning_rate,discount_rate,ep)
    rewards, steps = SarsaLearning_test(Q_table,no_of_demo)
    avg_reward = np.mean(rewards) 
    avg_steps = np.mean(steps)
    std_reward = np.std(rewards)
    std_steps = np.std(steps)
    
    return avg_reward,std_reward,avg_steps,std_steps

