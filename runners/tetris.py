import gymnasium as gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import random

# Environment setup
env = gym.make("ALE/Tetris-v5", obs_type="grayscale")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size, action_size)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
memory_size = 10000

# Experience Replay Memory
memory = []

# Build the Q-network
model = Sequential()
model.add(keras.Input(shape=(state_size,)))
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.summary()


#model.load_weights('tetris_gymnasium_weights.h5')

# Function to choose an action based on epsilon-greedy strategy
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    #print("predicting")
    #print(state)
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

    targets = rewards + gamma * (np.amax(model.predict_on_batch(next_states), axis=1)) * (1 - dones)
    target_values = model.predict_on_batch(states)
    target_values[np.arange(batch_size), actions] = targets

    model.fit(states, target_values, epochs=1)

# Training the agent
for episode in range(100):
    state = env.reset()
    # flatten
    #print(state)
    #print(state)
    total_reward = 0

    while True:
        # env.render()

        action = choose_action(state)
        observation, reward, terminated, truncated, info = env.step(action)

        if reward > 0:
            memory.append((state, action, reward, observation, terminated))
            if len(memory) > memory_size:
                memory.pop(0)

        total_reward += reward
        state = observation

        if terminated or truncated:
            train_network()
            print("Episode {}: Total Reward: {}, Epsilon: {:.2f}".format(episode, total_reward, epsilon))
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

model.save_weights('tetris_gymnasium_weights.h5')

# Close the environment
env.close()