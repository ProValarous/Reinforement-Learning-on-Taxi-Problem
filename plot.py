# epsilon = high, low, mid, decay
# learning_rate = high, low, mid
# discount = high, low , mid
# decay_rate = high, low

# avg reward vs no of training episodes

import matplotlib.pyplot as plt
import copy 
import time
from QLearning import QLearning
from SarsaLearning import SarsaLearning

class AnalysisPlot():
    def __init__(self) -> None:
        self.epsilon = [0.1,0.5,0.9]
        self.learning_rate = [0.1,0.5,0.9]
        self.discount_rate = [0.1,0.5,0.9]
       
        
        # decay, we'll see
        
    def QL_Reward_Ep(self,epsilon = 0.5,learning_rate = 0.9 ,discount_rate = 0.9,decay_rate=0,no_of_demo=10):
        no_of_episodes = [x for x in range(10,5000,100)]
        experiments = []
        
        # for i in self.epsilon:
        #     rewards = []
        #     for ep in no_of_episodes:
        #         rewards.append(QLearning(i,learning_rate,discount_rate,decay_rate,ep,no_of_demo)[0])
        #     experiments.append(copy.deepcopy(rewards)) 
        
        for i in self.learning_rate:
            rewards = []
            for ep in no_of_episodes:
                rewards.append(QLearning(epsilon,i,discount_rate,decay_rate,ep,no_of_demo)[0])
            experiments.append(copy.deepcopy(rewards))  
            
        # for i in self.discount_rate:
        #     rewards = []
        #     for ep in no_of_episodes:
        #         rewards.append(QLearning(epsilon,learning_rate,i,decay_rate,ep,no_of_demo)[0])
        #     experiments.append(copy.deepcopy(rewards))        
        
        
        plt.title("Q-Learning Analysis Plot")   
        plt.ylabel("Average Reward")
        plt.xlabel("Number of Training Episodes")

        plt.plot(no_of_episodes,experiments[0],label = "$\\alpha = 0.1$")
        plt.plot(no_of_episodes,experiments[1],label = "$\\alpha = 0.5$")
        plt.plot(no_of_episodes,experiments[2],label = "$\\alpha = 0.9$")
        
        # plt.plot(no_of_episodes,experiments[0],label = "$\epsilon = 0.1 $",linestyle = "--",color='red')
        # plt.plot(no_of_episodes,experiments[1],label = "$\epsilon = 0.5 $",linestyle = "-.",color='green')
        # plt.plot(no_of_episodes,experiments[2],label = "$\epsilon = 0.9 $",linestyle = "dotted",color='blue')
        
        # plt.plot(no_of_episodes,experiments[0],label = "$\\alpha = 0.1 $",linestyle = "--",color='red')
        # plt.plot(no_of_episodes,experiments[1],label = "$\\alpha = 0.5 $",linestyle = "-.",color='green')
        # plt.plot(no_of_episodes,experiments[2],label = "$\\alpha = 0.9 $",linestyle = "dotted",color='blue')
    
        # plt.plot(no_of_episodes,experiments[6],label = "$\gamma = 0.1 $",linestyle = "--",color='blue')
        # plt.plot(no_of_episodes,experiments[7],label = "$\gamma = 0.5 $",linestyle = "-.",color='blue')
        # plt.plot(no_of_episodes,experiments[8],label = "$\gamma = 0.9 $",linestyle = "dotted",color='blue')
        
        plt.legend()
        plt.show()

    def SarsaLearning_Reward_Ep(self,learning_rate=0.9,discount_rate=0.9,no_of_demo=10):
        no_of_episodes = [x for x in range(10,5000,100)]
        experiments = []
        # for i in [0.1,0.5,0.9]:
        for i in self.learning_rate:
            rewards = []
            for ep in no_of_episodes:
                rewards.append(SarsaLearning(i,discount_rate,ep,no_of_demo)[0])
            experiments.append(copy.deepcopy(rewards)) 

        plt.title("SARSA-Learning Analysis Plot")   
        plt.ylabel("Average Reward")
        plt.xlabel("Number of Training Episodes")

        plt.plot(no_of_episodes,experiments[0],label = "$\\alpha = 0.1$")
        plt.plot(no_of_episodes,experiments[1],label = "$\\alpha = 0.5$")
        plt.plot(no_of_episodes,experiments[2],label = "$\\alpha = 0.9$")
        plt.legend()
        plt.show()


        # rewards = []
        # for ep in no_of_episodes:
        #     rewards.append(SarsaLearning(epsilon,learning_rate,discount_rate,decay_rate,ep,no_of_demo)[0])
            
        # plt.title("SARSA-Learning Analysis Plot")
        # plt.ylabel("Average Reward")
        # plt.xlabel("Number of Training Episodes")
        # plt.plot(no_of_episodes,rewards)
        # plt.show()

    def Compartive_analysis_reward(self,epsilon=0.5,learning_rate=0.9,discount_rate=0.9,no_of_demo=10):
        no_of_episodes = [x for x in range(10,5000,100)]
        time_SL = []
        time_QL = []
        for ep in no_of_episodes:
            start_time = time.time()
            # rewards_QL.append(QLearning(epsilon,learning_rate,discount_rate,decay_rate,ep,no_of_demo)[0])
            QLearning(epsilon,learning_rate,discount_rate,ep,no_of_demo)
            end_time = time.time()
            time_QL.append(end_time-start_time)

            start_time = time.time()
            # rewards_SL.append(SarsaLearning(epsilon,learning_rate,discount_rate,decay_rate,ep,no_of_demo)[0])
            SarsaLearning(learning_rate,discount_rate,ep,no_of_demo)
            end_time = time.time()
            time_SL.append(end_time-start_time)

        plt.title("Comparative Analysis Plot")
        plt.ylabel("Average Reward")
        plt.xlabel("Number of Training Episodes")
        plt.plot(no_of_episodes,time_QL, label = "Q-learning", linestyle = "-")
        plt.plot(no_of_episodes,time_SL, label = "SARSA-learning", linestyle = "--")
        plt.legend()
        plt.show()
        
        
        
#####################
RLproject = AnalysisPlot()
# RLproject.QL_Reward_Ep()
# RLproject.SarsaLearning_Reward_Ep()
# RLproject.Compartive_analysis_reward(epsilon=0.5,learning_rate=0.8,discount_rate=0.9,decay_rate=0)    
RLproject.Compartive_analysis_reward()
        

