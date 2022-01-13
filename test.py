import rlcard
import torch
from rlcard.agents import RandomAgent
from bayes_agent import BAgent

env = rlcard.make('no-limit-holdem')
chips = [0, 0]
winrate = 0
count = 200000
threshold = 100000
bagent = BAgent(env.num_actions)
env.set_agents([RandomAgent(num_actions=env.num_actions), bagent])
for i in range(count):
    print(f"\rtrain epoch of part 1:", i , "chips:", chips ,end="")
    trajectories, player_wins = env.run(is_training=False)
    if i < threshold:
        bagent.memory.save(env.get_state(1), player_wins[1])
        bagent.memory.save(env.get_state(0), player_wins[0])
    if i >= threshold:
        chips[0] += player_wins[0]
        chips[1] += player_wins[1]
        if(player_wins[0] < 0 and player_wins[1] > 0):
            winrate += 1
        elif(player_wins[0] == 0 and player_wins[1] == 0):
            winrate += 1
winrate = winrate / (count - threshold)
print(chips[0], chips[1], winrate)