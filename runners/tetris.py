import gymnasium as gym
import numpy as np
import gym_totris
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Rescaling, LeakyReLU, InputLayer
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, CategoricalCrossentropy, Huber
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
import tensorflow as tf
import pickle
import pandas as pd
from collections import deque

demoMode = False
pretrainingMode = False

# Environment setup
env = gym.make("TOTRIS-v0")
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
global gamma
gamma = 0.99  # Discount factor
epsilon = 0.99  # Exploration-exploitation trade-off
epsilon_decay = 0.998
min_epsilon = 0.001
batch_size = 512
N_iteration = 1000

# Experience Replay Parameters
prioritized = True
buffer_size = 1e+6
# Beta linear annealing
global beta
beta = 0.4
beta_step = (1 - beta)/N_iteration

episodePerSave = 20
if demoMode:
    episodePerSave = 1

experimentName = "priority_smartreward_pretrained_normalized_leakyconv_reludense_linear_mse" # "smartreward_normalized_reludense_sigmoid_mse_131kmem_4kbatch"

loadMemoryFile = "memory/{}.pkl".format(experimentName)
saveMemoryFile = "memory/{}.pkl".format(experimentName)

loadWeightFile = "weights/{}.keras".format(experimentName)
saveWeightFile = "weights/{}.keras".format(experimentName)

loadResultsFile = "results/{}.pkl".format(experimentName)
saveResultsFile = "results/{}.pkl".format(experimentName)

results = deque()
try:
    with open(loadResultsFile, 'rb') as file:
        results = pickle.load(file)
        epsilon = results[-1]['epsilon']
except:
    print("Warning: no results loaded")

# Experience Replay Memory
def makeReplayBuffer():
    # https://ymd_h.gitlab.io/cpprb/examples/dqn_per/
    global gamma

    # Nstep
    nstep = 3
    # nstep = False

    if nstep:
        Nstep = {"size": nstep, "rew": "rew", "next": "next_obs"}
        gamma = tf.constant(gamma ** nstep)
    else:
        Nstep = None
        gamma = tf.constant(gamma)

    env_dict = {
        "obs":{"shape": env.observation_space['board'].shape},
        "act":{"shape": 1, "dtype": np.ubyte},
        "rew": {},
        "next_obs": {"shape": env.observation_space['board'].shape},
        "done": {}
    }

    if prioritized:
        rb = PrioritizedReplayBuffer(buffer_size, env_dict, Nstep=Nstep)
        # rb = PrioritizedReplayBuffer(buffer_size, env_dict)
        
        return rb
    else:
        rb = ReplayBuffer(buffer_size,env_dict, Nstep=Nstep)
        # rb = ReplayBuffer(buffer_size, env_dict)

        return rb

rb = makeReplayBuffer()
try:
    with open(loadMemoryFile, 'rb') as file:
        rb = pickle.load(file)
except:
    print("Warning: no memory loaded")

# Build the Q-network
model = Sequential()
#model.add(Rescaling(scale=1./7, input_shape=(20, 10, 1))) # [0-7] to [0-1]
model.add(InputLayer(input_shape=(20, 10, 1)))
model.add(Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01))) # input_shape=(20, 10, 1) here if not normalized
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(action_size))
model.compile(loss='huber', optimizer=Adam(learning_rate=learning_rate))

model.summary(show_trainable=True)

try:
    model.load_weights(loadWeightFile)
except:
    print("Warning: no weights loaded")

# Function to choose an action based on epsilon-greedy strategy
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict_on_batch(state)
    # assert model.predict(state).all() == q_values.all()
    return np.argmax(q_values[0])

# Function to train the Q-network using experience replay
def train_network():
    if rb.get_stored_size() < batch_size:
        return
    
    global beta
    global gamma

    if prioritized:
        mini_batch = rb.sample(batch_size, beta)
        beta += beta_step
    else:
        mini_batch = rb.sample(batch_size)

    states = mini_batch["obs"]
    actions = mini_batch["act"]
    rewards = mini_batch["rew"]
    next_states = mini_batch["next_obs"]
    dones = mini_batch["done"]

    target_Q = rewards + gamma * (tf.reduce_max(model.predict_on_batch(next_states), axis=1)) * (1 - dones)
    Q = model.predict_on_batch(states)
    TD_error = target_Q - Q[np.arange(batch_size), actions]
    Q[np.arange(batch_size), actions] = target_Q

    if prioritized:
        absTD = tf.math.abs(TD_error)
        rb.update_priorities(mini_batch["indexes"],absTD)

    model.fit(states, Q, epochs=1, verbose=1)

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
    for episode in range(1000):
        print("Pretraining: {}".format(episode))
        train_network()
    model.save_weights(saveWeightFile)

# Training the agent
if not pretrainingMode:
    episode = len(results)
    endEpisode = episode + N_iteration
    while episode < endEpisode:
        episode += 1
        observation = env.reset()
        observation = observation[0]['board']
        # Reshape the board to (batch_size, height, width, channels)
        observation = observation.reshape(1, observation.shape[0], observation.shape[1], 1)
        #observation = observation.reshape(1, observation.shape[0], observation.shape[1])
        observation[observation > 0] = 1

        total_reward = 0
        steps = 0
        while True:
            # env.render()

            action = 0
            if demoMode:
                action = pretraining_action()
            else:
                action = choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            next_observation = next_observation['board']
            next_observation = next_observation.reshape(1, next_observation.shape[0], next_observation.shape[1], 1)
            #observation = observation.reshape(1, observation.shape[0], observation.shape[1])
            next_observation[next_observation > 0] = 1
            rb.add(obs=observation, act=action, rew=reward, next_obs=next_observation, done=(terminated or truncated))

            total_reward += reward
            observation = next_observation

            if terminated or truncated:
                train_network()
                print("Episode {}: Total Reward: {}, Epsilon: {:.2f}, Drawn Pieces: {}, Lines Cleared: {}".format(episode, total_reward, epsilon, info['drawn_pieces'], info['total_lines_cleared']))
                print("Memory size: {}".format(rb.get_stored_size()))
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
                        pickle.dump(rb, output)

                    with open(saveResultsFile, 'wb') as output:
                        pickle.dump(results, output)
                    
                    backend.clear_session()
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Close the environment
    env.close()
elif pretrainingMode:
    pretrain()
