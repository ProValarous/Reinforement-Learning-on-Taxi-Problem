# epsilon = high, low, mid, decay
# learning_rate = high, low, mid
# discount = high, low , mid
# decay_rate = high, low

# avg reward vs no of training episodes

import matplotlib.pyplot as plt
from QLearning import QLearning

class AnalysisPlot():
    def __init__(self) -> None:
        self.epsilon_large = '-r'
        self.epsilon_mid = '^r'
        self.epsilon_small = '*r'
        self.epsilon_decay = '--r'
        
        self.learning_rate = '-b'
        self.learning_rate = '^b'
        self.learning_rate = '*b'
        
        self.discount_rate = '-g'
        self.discount_rate = '^g'
        self.discount_rate = '*g'
        
        # decay, we'll see
        
    def QL_Reward_Ep(self,epsilon,learning_rate,discount_rate,decay_rate,no_of_demo=10):
        no_of_episodes = [x for x in range(10,10000,100)]
        rewards = []
        for ep in no_of_episodes:
            rewards.append(QLearning(epsilon,learning_rate,discount_rate,decay_rate,ep,no_of_demo)[0])
            
        plt.title("Q-Learning Analysis Plot")
        plt.ylabel("Average Reward")
        plt.xlabel("Number of Training Episodes")
        plt.plot(no_of_episodes,rewards)
        plt.show()
        
        
        
#####################
RLproject = AnalysisPlot()
RLproject.QL_Reward_Ep(epsilon=0.5,learning_rate=0.8,discount_rate=0.9,decay_rate=0)
    
        

