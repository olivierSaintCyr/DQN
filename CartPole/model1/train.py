import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
from collections import deque
import tqdm
import matplotlib.pyplot as plt
import agent

EPISODE = 30

EPSILON_INIT = 0.99
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.99975

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    print(env.action_space.shape)
    DQNAgent = agent.agent(env)

    epsilon = EPSILON_INIT
    episodeRewards = []

    for episode in tqdm.tqdm(range(1, EPISODE + 1)):
        episodeReward = 0
        step = 1
        
        currentState = env.reset()
        
        done = False
        while not done:
            env.render()
            if random.random() > epsilon:
                action = np.argmax(DQNAgent.getQs(currentState))
            else:
                action = random.randint(0, 1)
            
            nextState, reward, done, _ = env.step(action)
            
            episodeReward += reward
            step += 1
            
            DQNAgent.updateReplayMemory((currentState, action, reward, nextState, done))
            DQNAgent.train(step)
            
            currentState = nextState
        print("REWARD EPISODE ", episode, " : ", episodeReward)  
        episodeRewards.append(episodeReward)
        
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    plt.plot(episodeRewards)
    plt.show()


            

            