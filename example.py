import rlcard
import torch
from rlcard.agents import RandomAgent
from dqn_card import DQNAgent
from nfsp_card import NFSPAgent
from d3qn_agent import D3QNAgent


env = rlcard.make('no-limit-holdem')
d3qn_agent = D3QNAgent()
nfsp_agent = NFSPAgent(num_actions=5,state_shape=env.state_shape,
                     hidden_layers_sizes=[256,128,64],q_mlp_layers=[500,200,100])
dqn_agent = DQNAgent(num_actions=5,state_shape=env.state_shape,mlp_layers=[128,128,128])
random_agent = RandomAgent(num_actions=env.num_actions)
epoch1 = int(1e5)
epoch2 = int(1e6)
epoch3 = int(2e6)
chipsAll = []
winRate = []
minChip = []

#training part 1: against Random Agent.
env.set_agents([random_agent,d3qn_agent])

print(env.num_actions) # 5
print(env.num_players) # 2
print(env.state_shape) # [[54], [54]]
print(env.action_shape) # [None, None]

chips = [0,0]
wins = 0
minc = 1e9
for i in range(epoch1):    #num_epoch = 1e5, which could be ungraded.
    print(f"\rtrain epoch:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=True)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    for j in range(len(trajectories)):
        # print("player: ", j)
        if j == 1:  #select our D3QN Agent.
            for k in range(len(trajectories[j]) // 2):#store memory for each epoch.
                # print("state, action: ", k)
                state = trajectories[j][2*k]
                action = trajectories[j][2*k + 1]
                next_state = trajectories[j][2*k + 2]
                reward = payoffs[1] if env.is_over is True else 0
                #print("begin feed")
                d3qn_agent.feed(state, action, reward, next_state, env.is_over())
                #print("end feed")
            if i > 100 and i%5==0:  #sample memory.
                d3qn_agent.train()
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch1)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of the first training: ',winRate[0])
print('chips up to the first training: ',chips)
print('min chip of the first training: ',minChip[0])


#training part 2: against original DQN Agent.
env.set_agents([dqn_agent,d3qn_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch2):    #num_epoch = 1e6.
    print(f"\rtrain epoch:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=True)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    for j in range(len(trajectories)):
        if j == 1:  #select our D3QN Agent.
            for k in range(len(trajectories[j]) // 2):#store memory for each epoch.
                state = trajectories[j][2*k]
                action = trajectories[j][2*k + 1]
                next_state = trajectories[j][2*k + 2]
                reward = payoffs[1] if env.is_over is True else 0
                #print(next_state)
                d3qn_agent.feed(state, action, reward, next_state, env.is_over())
            if i > 100 and i%5==0:  #sample memory.
                d3qn_agent.train()
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch2)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of the second training: ',winRate[1])
print('chips up to the second training: ',chips)
print('min chip of the second training: ',minChip[1])


#training part 3: against NFSP Agent.
env.set_agents([nfsp_agent,d3qn_agent])
chips = [0,0]
wins = 0
minc = 1e9

for i in range(epoch3):    #num_epoch = 2e6.
    print(f"\rtrain epoch:", i , "chips:", chips ,end="")
    trajectories, payoffs = env.run(is_training=True)
    chips[0] += payoffs[0]
    chips[1] += payoffs[1]
    for j in range(len(trajectories)):
        if j == 1:  #select our D3QN Agent.
            for k in range(len(trajectories[j]) // 2):#store memory for each epoch.
                state = trajectories[j][2*k]
                action = trajectories[j][2*k + 1]
                next_state = trajectories[j][2*k + 2]
                reward = payoffs[1] if env.is_over is True else 0
                d3qn_agent.feed(state, action, reward, next_state, env.is_over())
            if i > 100 and i%5==0:  #sample memory.
                d3qn_agent.train()
    if payoffs[1]>0:
        wins += 1
    if chips[1]<minc:
        minc = chips[1]

winRate.append(wins/epoch2)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of the third training: ',winRate[2])
print('chips up to the third training: ',chips)
print('min chip of the third training: ',minChip[2])

state_dic = {'q_net':d3qn_agent.q_network.state_dict(), 'target_net': d3qn_agent.target_network.state_dict(), 'q_optimizer': d3qn_agent.q_network.optimizer.state_dict(), 'target_optimizer': d3qn_agent.target_network.optimizer.state_dict(), 'memory': d3qn_agent.memory.memory, 'prob': d3qn_agent.memory.prob, 'lr': d3qn_agent.memory.lr, 'counter': d3qn_agent.memory.counter}
torch.save(state_dic, 'nfsp_rlcard.pth.tar')

#print(agentnfsp.show_memory())
