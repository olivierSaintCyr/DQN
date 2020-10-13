import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
from collections import deque

REPLAY_MEMORY_SIZE = 50000
MINI_BATCH_SIZE = 64
UPDATE_EVERY = 5
MIN_REPLAY_SIZE = 5
DISCOUNT = 0.99

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

class agent:
    def __init__(self, env):
        # Input and outputs size of the model
        self.inputSize = (4)
        self.outputSize = 2       
        # Main model
        self.model = self.createModel(env)
        
        # Target model
        self.modelTarget = self.createModel(env)
        self.modelTarget.set_weights(self.model.get_weights())

        # Replay memory transition = [currentState, action, reward, nextState, done]
        self.replayMemory = deque(maxlen=REPLAY_MEMORY_SIZE)
    
        # Counter to used to update de counter
        self.counterUpdate = 0

    def createModel(self, env):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(32, input_shape = (4,), activation='relu'))
        #model.add(keras.layers.Lambda(lambda x: keras.backend.stop_gradient(x)))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(self.outputSize, activation='linear'))
        model.compile(optimizer='adam', loss= 'mean_squared_error', metrics=['accuracy'])
        return model
    
    def getQs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def updateReplayMemory(self, transition):
        self.replayMemory.append(transition)
    
    def train(self, step):
        
        if len(self.replayMemory) < MINI_BATCH_SIZE:
            return

        minibatch = random.sample(self.replayMemory, MINI_BATCH_SIZE)
        
        currentStatesBatch = np.array([transition[0] for transition in minibatch])
        currentQsBatch = self.model.predict(currentStatesBatch)

        nextStatesBatch = np.array([transition[3] for transition in minibatch])
        nextQsBatch = self.model.predict(nextStatesBatch)

        # Generation of X and Y to needed to generate the model.
        X = []
        Y = []

        for index, (currentState, action, reward, nextState, done) in enumerate(minibatch):
            if not done:
                qNextMax = np.max(self.getQs(nextState))
                newQ = reward + DISCOUNT * qNextMax
            else:
                newQ = reward
            
            currentQs = currentQsBatch[index]
            currentQs[action] =  newQ
            
            X.append(currentState)
            Y.append(currentQs)
            
        self.modelTarget.fit(np.array(X), np.array(Y), batch_size=MINI_BATCH_SIZE, shuffle=False, verbose=0)

        self.counterUpdate += 1
        if self.counterUpdate == UPDATE_EVERY:
            self.model.set_weights(self.modelTarget.get_weights())
            self.counterUpdate = 0

    

    