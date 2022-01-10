from typing import Dict

import torch
import torch.nn as nn
import numpy as np
import random
import collections
from collections import namedtuple

from torch.nn import Parameter

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])

global device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DQNAgent(object):
    def __init__(self,
                 batch_size=32,
                 alpha=0.01,
                 tao=0.5,
                 gamma = 0.9,
                 env = None,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000
                 ):
        self.use_raw = False #may not run???
        self.alpha = alpha
        self.betas = np.linspace(0.2,1,num=500000)
        self.tao = tao  #for update target network.
        self.gamma = gamma  #discount factor.
        self.memory = Memory(batch_size=batch_size,alpha=self.alpha,betas=self.betas)
        self.q_network = Network()
        self.target_network = Network()
        self.env = env
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size

    def step(self, state):
        legal_actions = list(state['legal_actions'].keys())
        state = self.modify_state(state)

        Q = self.q_network.Qvalue(state)[0]
        #print("debug: Q: ", Q)

        masked_q_values = -np.inf * np.ones(5, dtype=float)
        masked_q_values[legal_actions] = Q[legal_actions]

        epsilon = self.epsilons[min(self.memory.counter, self.epsilon_decay_steps-1)]
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(masked_q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action

    def eval_step(self, state):
        state = self.modify_state(state)

        Q = self.q_network.Qvalue(state)[0]
        masked_q_values = -np.inf * np.ones(5, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = Q[legal_actions]

        best_action = legal_actions.index(np.argmax(masked_q_values))

        return best_action

    def feed(self, state, action, reward, next_state, done):
        state = self.modify_state(state)
        next_state = self.modify_state(next_state)

        # step1: forward q_network, calculate q_j
        Q = self.q_network.Qvalue(state)[0]

        # step2: select best action, calculate best_next_action
        Q_next = self.q_network.Qvalue(next_state)[0]
        masked_q_values = -np.inf * np.ones(5, dtype=float)
        next_legal_actions = list(next_state['legal_actions'].keys())
        masked_q_values[next_legal_actions] = Q_next[next_legal_actions]

        best_next_action = np.argmax(masked_q_values) #may fault??

        # step3: calculate the max value of next state q_j+1
        Q_next_target = self.target_network.Qvalue(next_state)[0]

        TD = reward + self.gamma * Q_next_target[best_next_action]
        TD_error = Q[action] - TD

        self.memory.save(state, action, reward, next_state, next_legal_actions, done, TD_error)
        return

    def modify_state(self, state):
        return np.expand_dims(state['obs'], 0)

    def train(self):
        indexes, record = self.memory.sample()
        state_batch, action_batch, reward_batch, next_state_batch, next_legal_actions_batch, done_batch = record

        # step1: forward q_network, calculate q_j
        Q = self.q_network.Qvalue(state_batch)[0]

        # step2: select best action, calculate best_next_action
        Q_next = self.q_network.Qvalue(next_state_batch)[0]
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * 5 for i in next_legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(5*self.batch_size, dtype=float)
        legal_actions = Q_next.flatten()[legal_actions]
        masked_q_values[legal_actions] = Q_next[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, 5))

        print("masked_q_values: ",masked_q_values)

        best_actions = np.argmax(masked_q_values, axis=1) #may fault???
        #best_next_action = legal_actions.index(np.argmax(masked_q_values))

        # step3: calculate the max value of next state q_j+1
        Q_next_target = self.target_network.Qvalue(next_state_batch)[0]

        # step4: calculate TD_error
        TD = reward_batch + self.gamma * Q_next_target[np.arange(self.batch_size), best_actions]
        TD_errors = Q[action_batch] - TD
        self.memory.update_TDerror(indexes, TD_errors)

        # step5: backward update q_network
        self.q_network.update(state_batch, action_batch, TD)

        # step6: backward update target_network
        target_para = self.target_network.named_parameters()
        q_network_para = self.q_network.named_parameters()
        dict_q = collections.OrderedDict(q_network_para)

        for key, param in target_para:
            if key in q_network_para:
                dict_q[key].data.copy_(self.tao * dict_q[key].data + (1 - self.tao) * param.data)
        # new_para = self.tao * q_network_para + (1 - self.tao) * target_para
        self.target_network.load_state_dict(dict_q)



    # def select_action(self, state):
    #     Q = self.q_network.Qvalue(state)
    #     action_index = torch.argmax(Q, dim = 1)
    #     return action_index






class Memory(object):
    def __init__(self, batch_size, alpha, betas):
        self.capacity = 500000 #hyperparameter 10^5-10^6
        self.size = 0
        self.counter = 0
        self.batch_size = batch_size
        self.memory = []
        self.prob = []
        self.lr = []
        self.alpha = alpha #initial lr
        self.betas = betas #hyperparameter
        # self.   for method 2


    def save(self, state, action, reward, next_state, legal_actions, done, TDerror):
        if self.size == self.capacity:
            self.memory.pop(0)
            self.prob.pop(0)
            self.lr.pop(0)
            self.size -= 1

        transition = Transition(state, action, reward, next_state, legal_actions, done)
        self.memory.append(transition)

        e = random.uniform(1e-5, 1e-4)

        #method 1
        p = TDerror + e

        #method 2
        #

        self.prob.append(p)

        if self.counter < 500000:
            beta = self.betas[self.counter]
        else:
            beta = 1
        self.lr.append(self.alpha * (self.counter * p) ** (beta))

        self.counter += 1
        self.size += 1


    def sample(self):
        prob_sum = sum(self.prob)
        # r = np.random.choice(self.memory, self.batch_size, p = [x / prob_sum for x in self.prob])
        indexes = np.random.choice(self.size, self.batch_size, p = [x / prob_sum for x in self.prob])
        r = [self.memory[i] for i in indexes]
        return indexes, r

    def update_TDerror(self, indexes, tderrors):
        for i in indexes:
            e = random.uniform(1e-5, 1e-4)

            #method 1
            p = tderrors[str(i)] + e

            #method 2
            #

            self.prob[i] = p

            if self.counter < 500000:
                beta = self.betas[self.counter]
            else:
                beta = 1
            self.lr[i] = (self.lr[i] * (self.counter * p) ** (-beta))



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer_dims = [54, 256, 256, 256, 256]
        self.V = nn.Linear(256, 1, bias=True)
        self.A = nn.Linear(256, 5, bias=True) #num_actions = 5
        fc = []
        for i in range(len(self.layer_dims)-1):
            fc.append(nn.Linear(self.layer_dims[i], self.layer_dims[i+1], bias=True))
            fc.append(nn.ReLU())
        self.encoder = nn.Sequential(*fc)
        self.device = device
        self.learning_rate = 0.005
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)


    def forward(self,state):
        s = torch.from_numpy(state).float().to(self.device)
        x = self.encoder(s)
        V = self.V(x)
        A = self.A(x)
        avg_A = A.mean()
        Q = (V + (A - avg_A))
        return Q

    def Qvalue(self, state):
        with torch.no_grad():
            Q = self.forward(state)
        return Q

    def update(self,s,a,y):
        self.optimizer.zero_grad()
        self.train()
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        Q = self.Qvalue(s)[a]

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()
        self.eval()

        return batch_loss
