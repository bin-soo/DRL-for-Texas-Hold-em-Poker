from typing import Dict

import torch
import torch.nn as nn
import numpy as np
import random
import collections
from collections import namedtuple

from torch.nn import Parameter
import copy

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])

global device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class D3QNAgent(object):
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

        #print("debug1: Q: ", self.q_network.Qvalue(state))

        Q = self.q_network.Qvalue(np.array(state))

        masked_q_values = -np.inf * np.ones(5, dtype=float)
        #print("debug2: ", masked_q_values)
        #print("debug3: ", Q)
        masked_q_values[legal_actions] = Q[legal_actions]
        #print("debug4: ")
        epsilon = self.epsilons[min(self.memory.counter, self.epsilon_decay_steps-1)]
        probs = np.zeros(5, dtype=float)
        probs[legal_actions] = epsilon / len(legal_actions)
        # print("debug6: epsilon", epsilon)
        # print("debug6: masked_q_values: ", masked_q_values)
        # print("debug6: argmax: ", np.argmax(masked_q_values))
        best_action_idx = np.argmax(masked_q_values)
        probs[best_action_idx] += (1.0 - epsilon)
        # print("debug7: probs", probs)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        # print("debug7: action", action)
        return action

    def eval_step(self, state):
        origin_state = state
        state = self.modify_state(state)

        Q = self.q_network.Qvalue(state)
        masked_q_values = -np.inf * np.ones(5, dtype=float)
        legal_actions = list(origin_state['legal_actions'].keys())
        masked_q_values[legal_actions] = Q[legal_actions]

        best_action = np.argmax(masked_q_values)

        return best_action

    def feed(self, state, action, reward, next_state, done):
        origin_next_state = next_state
        # print("debug5: origin next state: ", next_state)
        # print("debug5: state before: ", state)
        state = self.modify_state(state)
        # print("debug5: state after: ", state)
        # print("debug5: next state before: ", next_state)
        next_state = self.modify_state(next_state)
        # print("debug5: next state after: ", next_state)

        # step1: forward q_network, calculate q_j
        #print("debug5: begin step1")
        Q = self.q_network.Qvalue(np.array(state))
        #print("debug5: finish step1")

        # step2: select best action, calculate best_next_action
        Q_next = self.q_network.Qvalue(np.array(next_state))
        masked_q_values = -np.inf * np.ones(5, dtype=float)
        next_legal_actions = list(origin_next_state['legal_actions'].keys())
        # print("debug5: ", next_legal_actions)
        # print("debug5: ", masked_q_values)
        # print("debug5: ", Q_next)
        masked_q_values[next_legal_actions] = Q_next[next_legal_actions]

        best_next_action = np.argmax(masked_q_values) #may fault??

        # step3: calculate the max value of next state q_j+1
        Q_next_target = self.target_network.Qvalue(np.array(next_state))

        TD = reward + self.gamma * Q_next_target[best_next_action]
        TD_error = Q[action] - TD

        self.memory.save(state, action, reward, next_state, next_legal_actions, done, TD_error)
        return

    def card2index(self,string):
        suit=string[0]
        num=string[1]
        index=0
        if suit=='H':
            index+=1*13
        elif suit=='D':
            index+=2*13
        elif suit=='C':
            index += 3 * 13
        if num=='T':
            num=10
        elif num=='J':
            num=11
        elif num=='Q':
            num=12
        elif num=='K':
            num=13
        elif num=='A':
            num=1
        else:
            num=int(num)
        index+=(num-1)
        return index


    def modify_state(self, state):
        #print(state)
        l = copy.deepcopy(state['obs'])  # 长度54的手牌、公共牌信息
        for card in state['raw_obs']['hand']:
            i=self.card2index(card)
            if l[i]==0:
                print("there is something wrong!")
            l[i]=2
        temp=-1
        for i in range(len(l)-2):
            if i%13==0:
                temp=l[i]
            l[i]=l[i+1]
            if i%13==12:
                l[i]=temp
        #print("l:",l)
        #print(len(l))
        feature1 = 0
        feature2 = 0
        featureCard=[] #记录牌大小和花色
        for index, value in enumerate(l):
            if value == 2 and index < 52:
                feature1 += (index % 13)+1  # 手牌
                featureCard.append((index % 13)+1)
                featureCard.append(index // 13)
            if value == 1 and index < 52:
                feature2 += (index % 13)+1  # 公共牌
                featureCard.append((index % 13)+1)
                featureCard.append(index // 13)
        #print("feature1:",feature1)
        #print("feature2:",feature2)
        feature3 = len((state['raw_obs'])['public_cards'])  # 阶段
        #print("feature3:",feature3)
        feature4to7 = [0] * 4  # 花色
        for index, value in enumerate(l):
            if value != 0 and index < 52:
                #print(index // 13)
                feature4to7[index // 13] += 1
        #print("feature4to7:",feature4to7)
        # cardNum = [0] * 13
        # for index, value in enumerate(l):
        #     if value != 0 and index < 52:
        #         cardNum[index % 13] += 1
        while len(featureCard)!=14:
            featureCard.append(0)
        #print("featureCard:",featureCard)
        res=[]
        res.append(feature1)
        res.append(feature2)
        res.append(feature3)
        res+=feature4to7
        res+=featureCard
        #print(res)
        return res

    def train(self):
        indexes, record = self.memory.sample()
        #print(record)
        state_batch, action_batch, reward_batch, next_state_batch, next_legal_actions_batch, done_batch = record
        #print("\ndebug8: next_legal_actions_batch: ",next_legal_actions_batch)
        # step1: forward q_network, calculate q_j
        Q = self.q_network.Qvalue(state_batch)

        # step2: select best action, calculate best_next_action
        Q_next = self.q_network.Qvalue(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * 5 for i in next_legal_actions_batch[b]])
        #print("debug8: legal_actions: ", legal_actions)
        masked_q_values = -np.inf * np.ones(5*self.batch_size, dtype=float)
        # print("debug8: Q_next: ", Q_next)
        # print("debug8: legal_actions: ", legal_actions)
        masked_q_values[legal_actions] = Q_next.flatten().numpy()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, 5))

        #print("masked_q_values: ",masked_q_values)

        best_actions = np.argmax(masked_q_values, axis=1) #may fault???
        #best_next_action = legal_actions.index(np.argmax(masked_q_values))

        # step3: calculate the max value of next state q_j+1
        Q_next_target = self.target_network.Qvalue(next_state_batch)[0]

        # step4: calculate TD_error
        # print("\ndebug9: Q_next_target: ",Q_next_target)
        # print("debug9: ", Q_next_target.numpy()[best_actions])
        TD = reward_batch + self.gamma * Q_next_target.numpy()[best_actions]
        # print("\ndebug10: TD: ",TD)
        #print("debug10: action_batch: ", action_batch)
        action_batch_list = []
        for b in range(len(action_batch)):
            action_batch_list.extend([b * 5 + action_batch[b]])
        # print("debug10: action_batch_list: ", action_batch_list)
        # print("debug10: Q_f: ", Q.flatten())
        TD_errors = Q.flatten()[action_batch] - TD
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
        p = abs(TDerror) + e

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
        #prob_sum = sum(self.prob)
        # r = np.random.choice(self.memory, self.batch_size, p = [x / prob_sum for x in self.prob])
        #_p = [x.numpy() / prob_sum.numpy() for x in self.prob]
        _p = [x.numpy() for x in self.prob]
        prob_sum = sum(_p)
        _p /= prob_sum
        #print(sum(_p))
        indexes = np.random.choice(self.size, self.batch_size, p = _p)
        r = [self.memory[i] for i in indexes]
        r = map(np.array, zip(*r))
        return indexes, r

    def update_TDerror(self, indexes, tderrors):
        count = 0
        for i in indexes:
            e = random.uniform(1e-5, 1e-4)

            #method 1
            p = abs(tderrors[count]) + e
            count += 1

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
        self.layer_dims = [21, 256, 256, 256, 256]
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
        self.mse_loss = torch.nn.MSELoss(reduce=True, size_average=True)


    def forward(self,state):
        s = state
        if type(state) != torch.Tensor:
            s = torch.from_numpy(state).float().to(self.device)
        #print(s)
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
        if type(s) != torch.Tensor:
            s = torch.from_numpy(s).float().to(self.device)
        if type(a) != torch.Tensor:
            a = torch.from_numpy(a).long().to(self.device)
        if type(y) != torch.Tensor:
            y = torch.from_numpy(y).float().to(self.device)
        Q = self.Qvalue(s)
        # print("\ndebug11: Qvalue(s)", Q)
        action_index = []
        for b in range(len(a)):
            action_index.extend([a[b] + b * 5])
        Q = Q.flatten()[a]
        #print("\ndebug11: Q", Q)

        # update model
        batch_loss = self.mse_loss(Q, y)
        #print("\ndebug12: ", batch_loss)
        batch_loss = batch_loss.requires_grad_()
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()
        self.eval()

        return batch_loss
