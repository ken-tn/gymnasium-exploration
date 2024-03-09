import gymnasium as gym
import numpy as np
import gym_totris
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Rescaling, LeakyReLU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError
import random
import pickle
import pandas as pd
import tensorflow as tf
from collections import deque

# Environment setup
env = gym.make("TOTRIS-v0")
state_size = env.observation_space['board'].shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.99  # Exploration-exploitation trade-off
epsilon_decay = 0.998
min_epsilon = 0.01
batch_size = 4096
memory_size = 131072

episodePerSave = 10
experimentName = "rescale_linearconv_leakyrelu_softmax_mse_300kmem_16kbatch"

loadMemoryFile = "data/memory_{}.pkl".format(experimentName)
saveMemoryFile = "data/memory_{}.pkl".format(experimentName)

loadWeightFile = "training/weight_{}.keras".format(experimentName)
saveWeightFile = "training/weight_{}.keras".format(experimentName)

loadResultsFile = "data/results_{}.pkl".format(experimentName)
saveResultsFile = "data/results_{}.pkl".format(experimentName)

results = deque()
try:
    with open(loadResultsFile, 'rb') as file:
        results = pickle.load(file)
        epsilon = results[-1]['epsilon']
except:
    print("Warning: no results loaded")

# Experience Replay Memory
memory = deque(maxlen=memory_size)
try:
    with open(loadMemoryFile, 'rb') as file:
        oldMemory = pickle.load(file)
        # handle changing memory_size
        memory = [x for x in oldMemory]
except:
    print("Warning: no memory loaded")

# Build the Q-network
model = Sequential()
model.add(Rescaling(scale=1./7, input_shape=(20, 10, 1))) # [0-7] to [0-1]
model.add(Conv2D(32, (3, 3), activation='linear')) # input_shape=(20, 10, 1) here if not normalized
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='linear'))
model.add(Flatten())
model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
model.add(Dense(action_size, activation="softmax"))
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

model.summary(show_trainable=True)

try:
    model.load_weights(loadWeightFile)
except:
    print("Warning: no weights loaded")

# Function to choose an action based on epsilon-greedy strategy
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Function to train the Q-network using experience replay
def train_network():
    if len(memory) < batch_size:
        return

    mini_batch = random.sample(memory, batch_size)
    states = np.vstack([item[0] for item in mini_batch])
    actions = np.array([item[1] for item in mini_batch])
    rewards = np.array([item[2] for item in mini_batch])
    next_states = np.vstack([item[3] for item in mini_batch])
    dones = np.array([item[4] for item in mini_batch])

    targets = rewards + gamma * (np.amax(model.predict_on_batch(next_states), axis=1)) * (1 - dones) # Q(s, a)
    target_values = model.predict_on_batch(states) # 
    target_values[np.arange(batch_size), actions] = targets

    model.fit(states, target_values, epochs=1, verbose=1)

def pretraining_action():
    action = input()

    if action == "":
        return 1
    
    try:
        action = int(action)
        # enter = down, 1 = rotate, 2 = left, 3 = right
        actions = [1, 0, 2, 3]
        
        return actions[int(action)]
    except:
        return 1
    
def pretrain():
    for episode in range(2000):
        print("Pretraining: {}".format(episode))
        train_network()
    model.save_weights(saveWeightFile)

pretraining = False
#pretrain()

# Training the agent
if not pretraining:
    episode = len(results)
    while True:
        episode += 1
        state = env.reset()
        state = state[0]['board']
        # Reshape the board to (batch_size, height, width, channels)
        state = state.reshape(1, state.shape[0], state.shape[1], 1)

        total_reward = 0
        steps = 0
        while True:
            # env.render()

            #action = pretraining_action()
            action = choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            observation = observation['board']
            observation = observation.reshape(1, observation.shape[0], observation.shape[1], 1)

            if len(memory) >= memory_size:
                memory.pop(0)
            memory.append([state, action, reward, observation, terminated])

            total_reward += reward
            state = observation

            if terminated or truncated:
                train_network()
                print("Episode {}: Total Reward: {}, Epsilon: {:.2f}, Drawn Pieces: {}, Lines Cleared: {}".format(episode, total_reward, epsilon, info['drawn_pieces'], info['total_lines_cleared']))
                print("Memory size: {}".format(len(memory)))
                results.append(
                    {
                        'timestamp': pd.Timestamp.now(),
                        'episode': episode,
                        'total_reward': total_reward,
                        'epsilon': epsilon,
                        'drawn_pieces': info['drawn_pieces'],
                        'total_lines_cleared': info['total_lines_cleared'],
                        'total_tetris': info['total_tetris'],
                        'score': info['score'],
                        'steps': steps
                    }
                )

                if episode % episodePerSave == 0:
                    model.save_weights(saveWeightFile)
                    
                    with open(saveMemoryFile, 'wb') as output:
                        pickle.dump(memory, output)

                    with open(saveResultsFile, 'wb') as output:
                        pickle.dump(results, output)
                    
                    tf.keras.backend.clear_session()
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Close the environment
    env.close()
