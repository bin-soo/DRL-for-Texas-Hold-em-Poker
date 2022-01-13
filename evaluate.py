import rlcard
from rlcard import models
from rlcard.agents import NolimitholdemHumanAgent as HumanAgent
from rlcard.utils import print_card
from rlcard.agents import RandomAgent
from dqn_card import DQNAgent
from nfsp_card import NFSPAgent
from d3qn_agent import D3QNAgent
from bayes_agent import BAgent
import torch

# Make environment
# Set 'record_action' to True because we need it to print results
env = rlcard.make('no-limit-holdem', config={'record_action': True})

nfsp_agent = NFSPAgent(num_actions=5,state_shape=env.state_shape,
                     hidden_layers_sizes=[256,128,64],q_mlp_layers=[500,200,100])
dqn_agent = DQNAgent(num_actions=5,state_shape=env.state_shape,mlp_layers=[128,128,128])

random_agent = RandomAgent(num_actions=env.num_actions)

epoch1 = int(1e4)
epoch2 = int(1e4)
epoch3 = int(1e4)
epoch4 = int(1e4)
chipsAll = []
winRate = []
minChip = []

checkpoint = torch.load("d3qn001zg2.pth.tar")
d3qn_agent = D3QNAgent(epsilon_start=0, epsilon_end=0)
d3qn_agent.q_network.load_state_dict(checkpoint['q_net'])
d3qn_agent.target_network.load_state_dict(checkpoint['target_net'])
d3qn_agent.q_network.optimizer.load_state_dict(checkpoint['q_optimizer'])
d3qn_agent.target_network.optimizer.load_state_dict(checkpoint['target_optimizer'])
d3qn_agent.memory.memory = checkpoint['memory']
d3qn_agent.memory.prob = checkpoint['prob']
d3qn_agent.memory.lr = checkpoint['lr']
d3qn_agent.memory.counter = checkpoint['counter']

checkpoint = torch.load("NB001zg2.pth.tar")
b_agent = BAgent(num_actions=env.num_actions)
b_agent.memory.wincount=checkpoint['BA_wincount']
print(checkpoint['BA_wincount'])
b_agent.memory.losecount=checkpoint['BA_losecount']
print(checkpoint['BA_losecount'])
b_agent.memory.win=checkpoint['BA_win']
b_agent.memory.lose=checkpoint['BA_lose']
b_agent.count=checkpoint['BA_count']

env.set_agents([random_agent, b_agent])

chips = [0,0]
wins = 0
minc = 1e9
for i in range(epoch1):    #num_epoch = 1e5, which could be ungraded.
    print(f"\rtrain epoch of evaluate NB with random:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=False)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch1)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of evaluate NB with random: ',winRate[0])
print('chips up to evaluate NB with random: ',chips)
print('min chip of evaluate NB with random: ',minChip[0])


#training period 2: train BAgent.
env.set_agents([dqn_agent, b_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch2):
    print(f"\rtrain epoch of evaluate NB with dqn agent:", i, "chips:", chips, end="")
    trajectories, player_wins = env.run(is_training=False)
    chips[0] += player_wins[0]
    chips[1] += player_wins[1]
    if(player_wins[1] >= 0):
        wins += 1
    if chips[1] < minc:
        minc = chips[1]

winRate.append(wins / epoch2)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of evaluate NB with dqn agent: ',winRate[1])
print('chips up to evaluate NB with dqn agent: ',chips)
print('min chip of evaluate NB with dqn agent: ',minChip[1])

#training period 3: against BAgent.
env.set_agents([nfsp_agent, b_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch3):    #num_epoch = 1e5.
    print(f"\rtrain epoch of evaluate NB with nfsp agent:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=False)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch3)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of evaluate NB with nfsp agent: ',winRate[2])
print('chips up to evaluate NB with nfsp agent: ',chips)
print('min chip of evaluate NB with nfsp agent: ',minChip[2])

env.set_agents([random_agent, d3qn_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch1):    #num_epoch = 1e5, which could be ungraded.
    print(f"\rtrain epoch of evaluate d3qn with random:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=True)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch1)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of evaluate d3qn with random: ',winRate[3])
print('chips up to evaluate d3qn with random: ',chips)
print('min chip of evaluate d3qn with random: ',minChip[3])


#training period 2: train BAgent.
env.set_agents([dqn_agent, d3qn_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch2):
    print(f"\rtrain epoch of evaluate d3qn with dqn agent:", i, "chips:", chips, end="")
    trajectories, player_wins = env.run(is_training=True)
    chips[0] += player_wins[0]
    chips[1] += player_wins[1]
    if(player_wins[1] >= 0):
        wins += 1
    if chips[1] < minc:
        minc = chips[1]

winRate.append(wins / epoch2)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of evaluate d3qn with dqn agent: ',winRate[4])
print('chips up to evaluate d3qn with dqn agent: ',chips)
print('min chip of evaluate d3qn with dqn agent: ',minChip[4])

#training period 3: against BAgent.
env.set_agents([nfsp_agent, d3qn_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch3):    #num_epoch = 1e5.
    print(f"\rtrain epoch of evaluate d3qn with nfsp agent:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=True)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch3)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of evaluate d3qn with nfsp agent: ',winRate[5])
print('chips up to evaluate d3qn with nfsp agent: ',chips)
print('min chip of evaluate d3qn with nfsp agent: ',minChip[5])

#training period 3: against BAgent.
env.set_agents([b_agent, d3qn_agent])
chips = [0,0]
wins = 0
minc = 1e9
minc1 = 1e9

for i in range(epoch4):    #num_epoch = 1e5.
    print(f"\rtrain epoch of evaluate d3qn and NB:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=True)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]
    if chips[0]<minc1:
        minc1 = chips[0]

winRate.append(wins/epoch4)
chipsAll.append(chips)
minChip.append([minc1, minc])
print('win rate of d3qn agent: ',winRate[6])
print('chips up to d3qn agent: ',chips)
print('min chip of NB agent: ',minChip[6][0])
print('min chip of d3qn agent: ',minChip[6][1])