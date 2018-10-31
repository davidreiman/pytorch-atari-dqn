# Author: David Reiman
# github.com/davidreiman/PyTorch-Deep-Q-Learning

import os
import gym
import time
import copy
import random
import warnings
import numpy as np

import torch
import torchvision
import torch.nn as nn

from IPython import display
from collections import deque
from skimage.color import rgb2grey
from skimage.transform import rescale
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

plt.style.use('seaborn')
warnings.filterwarnings('ignore')


class DeepQNetwork(nn.Module):
    def __init__(self, num_frames, num_actions):
        super(DeepQNetwork, self).__init__()
        self.num_frames = num_frames
        self.num_actions = num_actions

        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=num_frames,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
            )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
            )
        self.fc1 = nn.Linear(
            in_features=3200,
            out_features=256,
            )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=num_actions,
            )

        # Activations
        self.relu = nn.ReLU()

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):

        # Forward pass
        x = self.relu(self.conv1(x)) # In: (80, 80, 4), Out: (20, 20, 16)
        x = self.relu(self.conv2(x)) # In: (20, 20, 16), Out: (10, 10, 32)
        x = self.flatten(x) # In: (10, 10, 32), Out: (3200,)
        x = self.relu(self.fc1(x)) # In: (3200,), Out: (256,)
        x = self.fc2(x) # In: (256,), Out: (4,)

        return x


class Agent:
    def __init__(self, model, memory_depth, lr, gamma, epsilon_i, epsilon_f, anneal_time):

        self.cuda = True if torch.cuda.is_available() else False
        self.to_tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.to_byte_tensor = torch.cuda.ByteTensor if self.cuda else torch.ByteTensor

        self.model = model

        if self.cuda:
            self.model = self.model.cuda()

        self.memory_depth = memory_depth
        self.gamma = self.to_tensor([gamma])
        self.e_i = epsilon_i
        self.e_f = epsilon_f
        self.anneal_time = anneal_time

        self.memory = deque(maxlen=memory_depth)
        self.clone()

        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def clone(self):
        self.clone_model = copy.deepcopy(self.model)

        for p in self.clone_model.parameters():
            p.requires_grad = False

    def remember(self, state, action, reward, terminal, next_state):

        if self.cuda:
            state, next_state = state.cpu(), next_state.cpu()

        state, next_state = state.data.numpy(), next_state.data.numpy()
        state, next_state = 255.*state, 255.*next_state
        state, next_state = state.astype(np.uint8), next_state.astype(np.uint8)
        self.memory.append([state, action, reward, terminal, next_state])

    def retrieve(self, batch_size):

        if batch_size > self.memories:
            batch_size = self.memories

        batch = random.sample(self.memory, batch_size)

        state = np.concatenate([batch[i][0] for i in range(batch_size)]).astype(np.int64)
        action = np.array([batch[i][1] for i in range(batch_size)], dtype=np.int64)[:, None]
        reward = np.array([batch[i][2] for i in range(batch_size)], dtype=np.int64)[:, None]
        terminal = np.array([batch[i][3] for i in range(batch_size)], dtype=np.int64)[:, None]
        next_state = np.concatenate([batch[i][4] for i in range(batch_size)]).astype(np.int64)

        state = self.to_tensor(state/255.)
        next_state = self.to_tensor(state/255.)
        reward = self.to_tensor(reward)
        terminal = self.to_byte_tensor(terminal)

        return state, action, reward, terminal, next_state

    @property
    def memories(self):
        return len(self.memory)

    def act(self, state):
        q_values = self.model(state).detach()

        if self.cuda:
            q_values = q_values.cpu()

        action = np.argmax(q_values.numpy())
        return action

    def process(self, state):
        state = rgb2grey(state[35:195, :, :])
        state = rescale(state, scale=0.5)
        state = state[np.newaxis, np.newaxis, :, :]
        return self.to_tensor(state)

    def exploration_rate(self, t):
        if 0 <= t < self.anneal_time:
            return self.e_i - t*(self.e_i - self.e_f)/self.anneal_time
        elif t >= self.anneal_time:
            return self.e_f
        elif t < 0:
            return self.e_i

    def huber_loss(self, x, y):
        error = x - y
        quadratic = 0.5 * error**2
        linear = np.absolute(error) - 0.5

        is_quadratic = (np.absolute(error) <= 1)

        return is_quadratic*quadratic + ~is_quadratic*linear

    def save(self, t, savedir=""):
        save_path = os.path.join(savedir, 'model-{}'.format(t))
        torch.save(self.model.state_dict(), save_path)

    def load(self, savedir):
        saves = [os.path.join(savedir, file) for file in os.listdir(savedir) if file.endswith('')]


    def update(self, batch_size, verbose=False):

        self.model.zero_grad()

        start = time.time()
        state, action, reward, terminal, next_state = self.retrieve(batch_size)

        if verbose:
            print("Sampled memory in {:0.2f} seconds.".format(time.time() - start))

        start = time.time()

        q = self.model(state)[range(batch_size), action.flatten()][:, None]
        qmax = self.clone_model(next_state).max(dim=1)[0][:, None]

        nonterminal_target = reward + self.gamma*qmax
        terminal_target = reward

        target = terminal.float()*terminal_target + (~terminal).float()*nonterminal_target

        loss = self.loss(q, target)

        loss.backward()
        self.opt.step()

        if verbose:
            print("Updated parameters in {:0.2f} seconds.".format(time.time() - start))


def q_iteration(episodes, plot=True, render=True, verbose=False):

    t = 0
    metadata = dict(episode=[], reward=[])

    try:
        progress_bar = tqdm(range(episodes), unit='episode')

        for episode in progress_bar:

            state = env.reset()
            state = agent.process(state)

            done = False
            total_reward = 0

            while not done:

                if render:
                    env.render()

                while state.size()[1] < num_frames:
                    action = np.random.choice(num_actions)

                    new_frame, reward, done, info = env.step(action)
                    new_frame = agent.process(new_frame)

                    state = torch.cat([state, new_frame], 1)

                if np.random.uniform() < agent.exploration_rate(t-burn_in) or t < burn_in:
                    action = np.random.choice(num_actions)

                else:
                    action = agent.act(state)

                new_frame, reward, done, info = env.step(action)
                new_frame = agent.process(new_frame)

                new_state = torch.cat([state, new_frame], 1)
                new_state = new_state[:, 1:, :, :]

                agent.remember(state, action, reward, done, new_state)

                state = new_state
                total_reward += reward
                t += 1

                if t % update_interval == 0 and t > burn_in:
                    agent.update(batch_size, verbose=verbose)

                if t % clone_interval == 0 and t > burn_in:
                    agent.clone()

                if t % save_interval == 0 and t > burn_in:
                    agent.save(t)

                if t % 1000 == 0:
                    progress_bar.set_description("t = {}".format(t))


            metadata['episode'].append(episode)
            metadata['reward'].append(total_reward)

            if episode % 100 == 0:
                avg_return = np.mean(metadata['reward'][-100:])
                print("Average return (last 100 episodes): {}".format(avg_return))

            if plot:
                plt.scatter(metadata['episode'], metadata['reward'])
                plt.xlim(0, episodes)
                plt.xlabel("Episode")
                plt.ylabel("Return")
                display.clear_output(wait=True)
                display.display(plt.gcf())

        return metadata

    except KeyboardInterrupt:
        print("Saving model before quitting...")
        agent.save(t)

        return metadata

# Hyperparameters

batch_size = 64
update_interval = 4
clone_interval = 128
save_interval = int(1e5)
frame_skip = None
num_frames = 4
num_actions = 4
episodes = int(1e4)
memory_depth = int(1e5)
epsilon_i = 1.0
epsilon_f = 0.1
anneal_time = int(1e6)
burn_in = int(1e4)
gamma = 0.99
learning_rate = 5e-4


model = DeepQNetwork(num_frames, num_actions)
agent = Agent(model, memory_depth, learning_rate, gamma, epsilon_i, epsilon_f, anneal_time)
env = gym.make('Breakout-v0')

metadata = q_iteration(episodes, plot=False, render=False)
