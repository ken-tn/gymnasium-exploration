import gymnasium as gym
import numpy as np
import gym_totris
from keras import backend
from keras.models import Sequential, clone_model, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Rescaling, LeakyReLU, Input, concatenate
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
min_epsilon = 0.05
batch_size = 256
N_iteration = 10050
target_update_freq = 500

# Experience Replay Parameters
prioritized = True
buffer_size = 1e+6
# Beta linear annealing
global beta
beta = 0.4
beta_step = (1 - beta)/N_iteration

episodePerSave = 60
if demoMode:
    episodePerSave = 1

experimentName = "correctedpos_priority_simulatednonperfectheuristicreward_DDQN_simpleinput_dense256relu_dense256relu_dense128relu_huber_256batch_pretrain"

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

    envshape = (22, )
    env_dict = {
        "obs":{"shape": envshape},
        "act":{"shape": 1, "dtype": np.ubyte},
        "rew": {},
        "next_obs": {"shape": envshape},
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
        rb.load_transitions(file)
        print("Loaded {} replay entries".format(rb.get_stored_size()))
except:
    print("Warning: no memory loaded")

optimizer=Adam(learning_rate=learning_rate)

@tf.function
def Huber_loss(absTD):
    return tf.where(absTD > 1.0, absTD, tf.math.square(absTD))

@tf.function
def MSE(absTD):
    return tf.math.square(absTD)

loss_func = Huber_loss

# Build the Q-network
# Define input layers for each component of the observation space
#board_input = Input(shape=(20, 10, 1), name='board')
#next_piece_input = Input(shape=(7,), name='next_piece')
#score_input = Input(shape=(1,), name='score')
input_layer = Input(shape=(22, ), name='inputlayer')

# Add convolutional layers
# rescaled_board = Rescaling(scale=(1./7), input_shape=(20, 10, 1))(board_input)
# conv1 = Conv2D(32, (8, 4), activation='elu', padding='same', strides=(4, 2))(rescaled_board)
# conv2 = Conv2D(64, (4, 2), activation='elu', padding='same', strides=(2, 1))(conv1)
# conv3 = Conv2D(64, (2, 1), activation='elu', padding='same')(conv2)

# Flatten the convolutional layer output
# flattened_conv = Flatten()(conv3)
# flattened_board = Flatten()(board_input)

# Concatenate the flattened convolutional layer with additional inputs
# concatenated_inputs = concatenate([flattened_conv, next_piece_input, score_input])
# concatenated_inputs = concatenate([flattened_board, next_piece_input, score_input])

# Add dense layers
dense1 = Dense(256, activation='relu')(input_layer)
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)

# Output layer
output = Dense(action_size)(dense3)

# Define the model
model = Model(inputs=[input_layer], outputs=output)
model.summary()

try:
    model.load_weights(loadWeightFile)
except:
    print("Warning: no weights loaded")

target_model = clone_model(model)

# Function to choose an action based on epsilon-greedy strategy
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = tf.squeeze(model(state.reshape(1,-1)))
    return np.argmax(q_values)

@tf.function
def Huber_loss(absTD):
    return tf.where(absTD > 1.0, absTD, tf.math.square(absTD))

@tf.function
def Q_func(model,obs,act,act_shape):
    return tf.reduce_sum(model(obs) * tf.one_hot(act,depth=act_shape), axis=1)

@tf.function
def DQN_target_func(model,target,next_obs,rew,done,gamma,act_shape):
    return gamma*tf.reduce_max(target(next_obs),axis=1)*(1.0-done) + rew

@tf.function
def Double_DQN_target_func(model,target,next_obs,rew,done,gamma,act_shape):
    """
    Double DQN: https://arxiv.org/abs/1509.06461
    """
    act = tf.math.argmax(model(next_obs),axis=1)
    return gamma*tf.reduce_sum(target(next_obs)*tf.one_hot(act,depth=act_shape), axis=1)*(1.0-done) + rew

target_func = Double_DQN_target_func
# Function to train the Q-network using experience replay
def train_network():
    if rb.get_stored_size() < batch_size:
        return
    
    global beta
    global gamma

    if prioritized:
        sample = rb.sample(batch_size, beta)
        beta += beta_step
    else:
        sample = rb.sample(batch_size)

    weights = sample["weights"].ravel() if prioritized else tf.constant(1.0)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        Q =  Q_func(model,
                    tf.constant(sample["obs"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(action_size, dtype="int32"))
        target_Q = target_func(model,target_model,
                               tf.constant(sample['next_obs']),
                               tf.constant(sample["rew"].ravel()),
                               tf.constant(sample["done"].ravel()),
                               gamma,
                               tf.constant(action_size, dtype="int32"))
        absTD = tf.math.abs(target_Q - Q)
        loss = tf.reduce_mean(loss_func(absTD)*weights)

    grad = tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))
    
    print("Loss: {}".format(loss))

    if prioritized:
        Q = Q_func(model,
                tf.constant(sample["obs"]),
                tf.constant(sample["act"].ravel()),
                tf.constant(action_size, dtype="int32"))
        absTD = tf.math.abs(target_Q - Q)
        rb.update_priorities(sample["indexes"], absTD)

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

# Training the agent
if not pretrainingMode:
    episode = len(results)
    while episode < N_iteration:
        episode += 1
        total_reward = 0
        steps = 0
        # norewcounter = 0

        observation, info = env.reset()
        # use next_piece, true_holes, tuck_setup_holes, surface, average_height, current_piece_positions as inputs
        # {'drawn_pieces': 1, 'total_lines_cleared': 0, 'total_tetris': 0, 'true_holes': 0, 'tuck_setup_holes': 0, 'surface': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'average_height': 0,
        # 'current_piece_positions': [{'X': 4, 'Y': 0}, {'X': 5, 'Y': 0}, {'X': 4, 'Y': 1}, {'X': 5, 'Y': 1}]}
        
        observation = np.concatenate([
            observation["next_piece"],
            [info["true_holes"]],
            [info["tuck_setup_holes"]],
            [info["average_height"]],
            info["surface"],
            [value for pos in info['current_piece_positions'] for value in pos.values()]
        ])
        observation = observation.reshape((1, observation.shape[0]))
        while True:
            # env.render()

            action = 0
            if demoMode:
                action = pretraining_action()
            else:
                action = choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            # if reward <= 0:
            #     norewcounter += 1
            #     if norewcounter >= 40:
            #         truncated = True
            next_observation = np.concatenate([
                next_observation["next_piece"],
                [info["true_holes"]],
                [info["tuck_setup_holes"]],
                [info["average_height"]],
                info["surface"],
                [value for pos in info['current_piece_positions'] for value in pos.values()]
            ])
            next_observation = next_observation.reshape((1, next_observation.shape[0]))
            rb.add(obs=observation, act=action, rew=reward, next_obs=next_observation, done=(terminated or truncated))

            steps += 1
            total_reward += reward
            observation = next_observation

            if terminated or truncated:
                train_network()
                rb.on_episode_end()
                print("Episode {}: Total Reward: {}, Epsilon: {:.2f}, Drawn Pieces: {}, Lines Cleared: {}".format(episode, total_reward, epsilon, info['drawn_pieces'], info['total_lines_cleared']))
                print("Stored replay buffer: {}".format(rb.get_stored_size()))
                results.append(
                    {
                        'timestamp': pd.Timestamp.now(),
                        'episode': episode,
                        'total_reward': total_reward,
                        'epsilon': epsilon,
                        'drawn_pieces': info['drawn_pieces'],
                        'total_lines_cleared': info['total_lines_cleared'],
                        'total_tetris': info['total_tetris'],
                        'score': observation[-1],
                        'steps': steps
                    }
                )

                if episode % target_update_freq == 0:
                    target_model.set_weights(model.get_weights())

                if episode % episodePerSave == 0:
                    model.save_weights(saveWeightFile)
                    
                    with open(saveMemoryFile, 'wb') as output:
                        rb.save_transitions(output)

                    with open(saveResultsFile, 'wb') as output:
                        pickle.dump(results, output)
                    
                    backend.clear_session()
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Close the environment
    env.close()
elif pretrainingMode:
    pretrain()
