import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from SumTree import SumTree

BUFFER_SIZE = int(1e5)  #size of replay buffer
BATCH_SIZE = 80 # size of sample mini batches        
GAMMA = 0.99 # Discount rate
PRI_A = 0.95 # priority experience replay (PER) coefficient 'a'
PRI_EPSILON = 1e-2 # PER coefficient epsilon 
TAU = 1e-3 # used for soft update          
LR = 1e-4 #the learning rate used by all four NNs.      

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    # Initializes the agent and all four NNs.
    def __init__(self, state_size, action_size, seed):
        
        # Define state and action space size and random seed used in NNS
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Define the actor NNs
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

        # Define the critic NNs
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)

        #Define the replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
    
    #Called every timestep t in order to learn
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done, self) # Add experience to buffer
        
        # Tell the agent to learn when there are enough experiences in the buffer in order to create a mini batch
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    # Chooses an action for the agent to perform given a particular state
    def act(self, state):
        # Choose the best action
        self.actor_local.eval()
        with torch.no_grad():
            action = np.clip(self.actor_local(torch.from_numpy(state).float().to(device)).cpu().data.numpy(), -1, 1)
        self.actor_local.train()
        
        return action

    # Updates the parameters of all four NNs every timestep t
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences # Get experiences from the replay buffer
        
        # updates weights for critic network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
 
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets) # Calculates critic loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # updates weights for actor network
        actions_actor = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_actor).mean() # Calculates actor loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # applies weights to NNs using a soft update rule
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    # Updates weights in NN using soft update rule
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    
    # Initializes values for replay buffer
    def __init__(self, buffer_size, batch_size):
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.replay_buffer_Tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size;
        self.buffer_count = 0
    
    # Adds value to buffer using loss of actor network as p
    def add(self, state, action, reward, next_state, done, agent):
        e = self.experience(state, action, reward, next_state, done)
        
        states = torch.from_numpy(state).float().to(device)
        
        # calculate actor loss
        agent.actor_local.eval()
        agent.critic_local.eval()
        with torch.no_grad():
            actions_pred = agent.actor_local(states)
            actor_loss = -agent.critic_local(states, actions_pred).mean()
        agent.actor_local.train()
        agent.critic_local.train()
        
        # add values to buffer
        p = (np.abs(actor_loss)+PRI_EPSILON)**PRI_A # calculate p using actor loss
        self.replay_buffer_Tree.add(p, e) # add experience to buffer using priority p
        self.buffer_count = min(self.buffer_size, self.buffer_count+1) # update buffer size counter
    
    # Samples the buffer for samples with high priority
    def sample(self):
        # samples enough experiences to create mini batch
        experiences = [self.replay_buffer_Tree.get(s) for s in np.random.uniform(0,self.replay_buffer_Tree.total(),self.batch_size)]
        for e in experiences:
            self.replay_buffer_Tree.update(e[0], e[1]*0.99)
        
        # creates lists of states actions rewards and dones
        states = torch.from_numpy(np.vstack([e[2].state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[2].action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2].reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[2].next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[2].done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    # Returns the current length of the buffer
    def __len__(self): 
        return self.buffer_count