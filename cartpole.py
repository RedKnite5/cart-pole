import math
import random
import os
from collections import namedtuple, deque
from itertools import count

import numpy as np

from PIL import Image

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib
import matplotlib.pyplot as plt





Load = False
dir = "save"

env = gym.make('CartPole-v0').unwrapped


print(env.observation_space)


plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        
        self.n_inputs = env.observation_space.shape[0]
        
        linear_size = 16
        
        self.start = nn.Linear(self.n_inputs * 1, linear_size)
        self.head = nn.Linear(linear_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.start(x))
        x = F.relu(self.head(x))
        
        return x


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART



env.reset()
#plt.figure()
#plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#           interpolation='none')
#plt.title('Example extracted screen')
#plt.show()


# Get number of actions from gym action space
n_actions = env.action_space.n





BATCH_SIZE = 64 # default 128
GAMMA = 0.999
EPS_START = 0.95  # default .9
EPS_END = 0.05   # default .05
EPS_DECAY = 200  # default 200
TARGET_UPDATE = 10 # default 10
num_episodes = 100
MEM_CAP = 100_000

if Load:
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    try:
        policy_net_dict = torch.load(os.path.join(dir, "net.pt"))
        
        policy_net = DQN(n_actions)
        target_net = DQN(n_actions)
        
        policy_net.load_state_dict(policy_net_dict)
        target_net.load_state_dict(policy_net_dict)
        target_net.eval()
    except FileNotFoundError:
        print("using default net")
        policy_net = DQN(n_actions).to(device)
        target_net = DQN(n_actions).to(device)
        
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    
    try:
        memory = torch.load(os.path.join(dir, "memory.pt"))
    except FileNotFoundError:
        print("using default memory")
        memory = ReplayMemory(MEM_CAP)
        
    try:
        steps_done = torch.load(os.path.join(dir, "steps.pt"))
        old_steps = steps_done
    except FileNotFoundError:
        print("no steps")
        steps_done = 0
        
    try:
        episode_durations = torch.load(os.path.join(dir, "durations.pt"))
    except FileNotFoundError:
        print("no durations")
        episode_durations = []

else:
    print("using default net")
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    print("using default memory")
    memory = ReplayMemory(MEM_CAP)
    
    print("no steps")
    steps_done = 0
    
    print("no durations")
    episode_durations = []




optimizer = optim.RMSprop(policy_net.parameters())

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:  # I set the hard cutoff
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state)
            print(f"{action.shape=}")
            return action.max(0)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration (frames)')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


try:
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = torch.tensor(env.reset(), dtype=torch.float)
        for t in count():
            # Select and perform an action
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float)
            reward = torch.tensor([reward], device=device)
            
            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                #print("Episode =", i_episode, end="\r")
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0 and i_episode:
            target_net.load_state_dict(policy_net.state_dict())
            
except:
    raise
finally:
    if Load:
        print("\nSaving")
        torch.save(policy_net.state_dict(), os.path.join(dir, "net.pt"))
        torch.save(memory, os.path.join(dir, "memory.pt"))
        torch.save(steps_done, os.path.join(dir, "steps.pt"))
        torch.save(episode_durations, os.path.join(dir, "durations.pt"))


print('Complete')
env.render()
env.close()




