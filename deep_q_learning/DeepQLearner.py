import unittest
import random
import json
import numpy as np
import gym
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras.optimizers import Adam


class DeepQLearner:
    # initialize replay memory D
    # initialize action-value function Q with random weights
    # observe initial state s
    # repeat
    #     select an action a
    #         with probability ε select a random action
    #         otherwise select a = argmaxa’Q(s,a’)
    #     carry out action a
    #     observe reward r and new state s’
    #     store experience <s, a, r, s’> in replay memory D
    #
    #     sample random transitions <ss, aa, rr, ss’> from replay memory D
    #     calculate target for each minibatch transition
    #         if ss’ is terminal state then tt = rr
    #         otherwise tt = rr + γmaxa’Q(ss’, aa’)
    #     train the Q network using (tt - Q(ss, aa))^2 as loss
    #
    #     s = s'
    # until terminated

    def __init__(self, original_model, environment, input_shape, max_memory_size=500, epsilon=.1, epsilon_decay=.95, discount=.9, sample_size=50):
        self.Q_model = original_model
        self.action_space = environment.action_space
        self.environment = environment

        self.input_shape = input_shape

        self.memories = list()
        self.max_memory_size = max_memory_size

        self.state = None

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.sample_size = sample_size

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.predict_reward(state))

    def act(self, action):
        return self.environment.step(action)

    def store_memory(self, state, action, reward, next_state, done):
        self.memories.append((state, action, reward, next_state, done))
        if len(self.memories) > self.max_memory_size:
            del self.memories[0]

    def sample_actions(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(self.memories, min(len(self.memories), self.sample_size))

        for state, action, reward, next_state, done in mini_batch:
            y_target = self.predict_reward(state)
            y_target[action] = reward if done else reward + self.discount * np.max(self.predict_reward(next_state))
            x_batch.append(state)
            y_batch.append(y_target)

        return np.array(x_batch), np.array(y_batch)

    def predict_reward(self, state):
        return self.Q_model.predict(np.reshape(state, self.input_shape))[0]

    def train_network(self):
        inputs, targets = self.sample_actions()
        return self.Q_model.train_on_batch(inputs, targets)

    def run(self, num_epochs):
        # Train
        for e in range(num_epochs):
            loss = 0.

            game_over = False
            # get initial input
            input_t = self.environment.reset()

            cumulative_reward = 0.
            true_reward = 0.

            num_steps = 0

            while not game_over:
                input_tm1 = input_t
                # get next action
                action = self.select_action(input_tm1)
                # print("Selected action:{}", action)

                # apply action, get rewards and new state
                input_t, reward, game_over, _ = self.act(action)
                self.environment.render()

                true_reward += reward

                if e > 50:
                    time_punishment = np.floor(np.power(num_steps / 200, 1.2))
                    reward -= time_punishment

                # store experience
                self.store_memory(input_tm1, action, reward, input_t, game_over)

                self.train_network()

                cumulative_reward += reward

                num_steps += 1

            print("Epoch {:03d}/{} | Steps {} | Reward {:4f} | True Reward {:4f}"
                  .format(e, num_epochs - 1, num_steps, cumulative_reward, true_reward))
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    epsilon = .7  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 1000
    max_memory = 500000
    hidden_size = 100
    batch_size = 50
    grid_size = 10
    discount = .9

    env = gym.make('LunarLander-v2')

    tmp_state = env.reset()

    model = Sequential()
    model.add(Dense(24, input_shape=env.observation_space.shape, activation='tanh'))
    model.add(Dense(48, activation='tanh'))
    model.add(Dense(4, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # Define environment/game
    # env = Catch(grid_size)

    learner = DeepQLearner(model, env, [1, 8], epsilon=epsilon, max_memory_size=max_memory
                           , sample_size=batch_size, discount=discount)

    learner.run(num_epochs=epoch)

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    print("Saved models")
