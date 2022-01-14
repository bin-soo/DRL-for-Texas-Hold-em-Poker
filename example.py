import rlcard
import torch
from rlcard.agents import RandomAgent
from dqn_card import DQNAgent
from nfsp_card import NFSPAgent
from d3qn_agent import D3QNAgent
from bayes_agent import BAgent
'''
every time you want to run this file, please rewrite the name of the path 
to make sure you have save every model instead of overwritting.
'''
path1 = 'd3qn001.pth.tar'     #next time you run this file, you can change path to 'd3qn04.pth.tar' for example.
path2 = 'NB001.pth.tar'
env = rlcard.make('no-limit-holdem')
d3qn_agent = D3QNAgent()
b_agent = BAgent(num_actions=env.num_actions)
nfsp_agent = NFSPAgent(num_actions=5,state_shape=env.state_shape,
                     hidden_layers_sizes=[256,128,64],q_mlp_layers=[500,200,100])
dqn_agent = DQNAgent(num_actions=5,state_shape=env.state_shape,mlp_layers=[128,128,128])
random_agent = RandomAgent(num_actions=env.num_actions)
epoch1 = int(1e5)
epoch2 = int(5e5)
epoch3 = int(1e5)
chipsAll = []
winRate = []
minChip = []
def save_dqndic(q_net, target_net, q_optimizer, target_optimizer, memory, prob, lr, counter):
    state_dic = {'q_net':q_net,
                 'target_net': target_net,
                 'q_optimizer': q_optimizer,
                 'target_optimizer': target_optimizer,
                 'memory': memory,
                 'prob': prob,
                 'lr': lr,
                 'counter': counter
                 }
    return state_dic

def save_NBdic(BA_wincount, BA_losecount, BA_win, BA_lose, BA_count):
    state_dic = {'BA_wincount':BA_wincount,
                 'BA_losecount':BA_losecount,
                 'BA_win':BA_win,
                 'BA_lose':BA_lose,
                 'BA_count':BA_count}
    return state_dic

#training period 1: against Random Agent.
env.set_agents([random_agent,d3qn_agent])

print(env.num_actions) # 5
print(env.num_players) # 2
print(env.state_shape) # [[54], [54]]
print(env.action_shape) # [None, None]

chips = [0,0]
wins = 0
minc = 1e9
for i in range(epoch1):    #num_epoch = 1e5, which could be ungraded.
    print(f"\rtrain epoch of part 1:", i , "chips:", chips ,end="")
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
    if i%10000==0:
        state_dic = save_dqndic(d3qn_agent.q_network.state_dict(),
                 d3qn_agent.target_network.state_dict(),
                 d3qn_agent.q_network.optimizer.state_dict(),
                 d3qn_agent.target_network.optimizer.state_dict(),
                 d3qn_agent.memory.memory,
                 d3qn_agent.memory.prob,
                 d3qn_agent.memory.lr,
                 d3qn_agent.memory.counter)
        torch.save(state_dic, path1)

winRate.append(wins/epoch1)
chipsAll.append(chips)
minChip.append(minc)
print('win rate of the first training: ',winRate[0])
print('chips up to the first training: ',chips)
print('min chip of the first training: ',minChip[0])


#training period 2: train BAgent.
env.set_agents([random_agent, b_agent])
chips = [0,0]
wins = 0
minc = 1e9
threshold = 1e5

for i in range(epoch2):
    print(f"\rtrain epoch of part 2:", i, "chips:", chips, end="")
    trajectories, player_wins = env.run(is_training=False)
    b_agent.memory.save(env.get_state(1), player_wins[1])
    if i < threshold:
        b_agent.memory.save(env.get_state(1), player_wins[1])
        b_agent.memory.save(env.get_state(0), player_wins[0])
    else:
        chips[0] += player_wins[0]
        chips[1] += player_wins[1]
        if(player_wins[1] >= 0):
            wins += 1
        if chips[1] < minc:
            minc = chips[1]
    if i%10000==0:
        state_dic = save_NBdic(b_agent.memory.wincount,
                 b_agent.memory.losecount,
                 b_agent.memory.win,
                 b_agent.memory.lose,
                 b_agent.count)
        torch.save(state_dic, path2)

winRate.append(wins / (epoch2 - threshold))
chipsAll.append(chips)
minChip.append(minc)
print('win rate of the bagent training: ',winRate[1])
print('chips up to the bagent training: ',chips)
print('min chip of the bagent training: ',minChip[1])
#
# #training period 3: against BAgent.
# env.set_agents([b_agent,d3qn_agent])
# chips = [0,0]
# wins = 0
# minc = 1e9
#
# for i in range(epoch3):    #num_epoch = 1e5.
#     print(f"\rtrain epoch of part 3:", i , "chips:", chips ,end="")
#     trajectories, payoffs = env.run(is_training=True)
#     chips[0] += payoffs[0]
#     chips[1] += payoffs[1]
#     for j in range(len(trajectories)):
#         if j == 1:  #select our D3QN Agent.
#             for k in range(len(trajectories[j]) // 2):#store memory for each epoch.
#                 state = trajectories[j][2*k]
#                 action = trajectories[j][2*k + 1]
#                 next_state = trajectories[j][2*k + 2]
#                 reward = payoffs[1] if env.is_over is True else 0
#                 #print(next_state)
#                 d3qn_agent.feed(state, action, reward, next_state, env.is_over())
#             if i > 100 and i%5==0:  #sample memory.
#                 d3qn_agent.train()
#     if payoffs[1]>0:
#         wins += 1
#     if chips[1]<minc:
#         minc = chips[1]
#     if i%10000==0:
#         state_dic = save_NBdic(b_agent.memory.wincount,
#                  b_agent.memory.losecount,
#                  b_agent.memory.win,
#                  b_agent.memory.lose,
#                  b_agent.count)
#         torch.save(state_dic, path2)
#
# winRate.append(wins/epoch3)
# chipsAll.append(chips)
# minChip.append(minc)
# print('win rate of the second training: ',winRate[2])
# print('chips up to the second training: ',chips)
# print('min chip of the second training: ',minChip[2])




#torch.save(state_dic, path)

#print(agentnfsp.show_memory())
