import gymnasium as gym
import numpy as np
import gym_totris
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import pickle

# Environment setup
env = gym.make("TOTRIS-v0")
state_size = env.observation_space['board'].shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.01  # Exploration-exploitation trade-off
epsilon_decay = 0.998
min_epsilon = 0.01
batch_size = 512
memory_size = 30000

loadMemoryFile = "memory_using_pretrained.pkl"
loadWeightFile = "tetris_gymnasium_weights_using_pretrained.h5"

saveMemoryFile = "memory_using_pretrained.pkl"
saveWeightFile = "tetris_gymnasium_weights_using_pretrained.h5"

episodePerSave = 10

# Experience Replay Memory
memory = []
try:
    pkl_file = open(loadMemoryFile, 'rb')
    memory = pickle.load(pkl_file)
    pkl_file.close()
except:
    print("Warning: no memory loaded")

# Build the Q-network
model = Sequential()
model.add(Dense(64, input_dim=state_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(action_size, activation='linear'))
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

    model.fit(states, target_values, epochs=1, verbose=0)

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

# Training the agent
episode = 0
while True:
    episode += 1
    state = env.reset()
    state = state[0]['board']
    state = np.reshape(state, [1, state_size])

    total_reward = 0
    while True:
        # env.render()

        #action = pretraining_action()
        action = choose_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        observation = observation['board']
        observation = np.reshape(observation, [1, state_size])

        memory.append([state, action, reward, observation, terminated])
        if len(memory) > memory_size:
            memory.pop(0)

        total_reward += reward
        state = observation

        if terminated or truncated:
            train_network()
            print("Episode {}: Total Reward: {}, Epsilon: {:.2f}".format(episode, total_reward, epsilon))

            if episode % episodePerSave == 0:
                model.save_weights(saveWeightFile)

                output = open(saveMemoryFile, 'wb')
                pickle.dump(memory, output)
                output.close()
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Close the environment
env.close()
