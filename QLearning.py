import numpy as np
import gym
import random
import copy


def main():
    # create Taxi environment
    env_train = gym.make("Taxi-v3")

    # initialize q-table
    state_size = env_train.observation_space.n
    action_size = env_train.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99  # per episode
    episode = 1
    
    convergence_threshold_count = 0
    
    # training
    while(True):
        # reset the environment
        state, _ = env_train.reset()
        done = False
        q = copy.deepcopy(qtable)
        for s in range(max_steps):  
            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env_train.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :]) # <- wrong : this cannot be same as greedy policy!

            # take action and observe reward
            new_state, reward, done, info, _ = env_train.step(action)
                
            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            
                      
            
            # Update to our new state
            state = new_state

            # if done, finish episode
            if done==True:
                break
        
        if np.array_equal(qtable,q):
            convergence_threshold_count+=1  
        else:
            convergence_threshold_count=0        
        
        if convergence_threshold_count > 3:
            break
        
        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)
        episode+=1

    print(f"Training completed over {episode} episodes")
    input("Press Enter to watch trained agent...")

    env_train.close()

    ###################################################

    env_test = gym.make("Taxi-v3",render_mode='human')
    
    for i in range(10):
        state, _ = env_test.reset()
        done = False
        rewards = 0
        print("Demo # ",i+1)
        for s in range(max_steps):
            print(f"TRAINED AGENT")
            print("Step {}".format(s + 1))

            action = np.argmax(qtable[state, :])
            new_state, reward, done, info, _ = env_test.step(action)
            rewards += reward
            
            env_test.render()
            print(f"score: {rewards}")
            state = new_state

            if done == True:
                break
        print("#################")   
        
    env_test.close()

if __name__ == "__main__":
    main()
