import rlcard
from rlcard import models
from rlcard.agents import NolimitholdemHumanAgent as HumanAgent
from rlcard.utils import print_card
from dqn_card import DQNAgent
from nfsp_card import NFSPAgent
from d3qn_agent import D3QNAgent
from bayes_agent import BAgent
import torch

checkpoint = torch.load("d3qn02.pth.tar")
d3qn_agent = D3QNAgent(epsilon_start=0,epsilon_end=0)
d3qn_agent.q_network.load_state_dict(checkpoint['q_net'])
d3qn_agent.target_network.load_state_dict(checkpoint['target_net'])
d3qn_agent.q_network.optimizer.load_state_dict(checkpoint['q_optimizer'])
d3qn_agent.target_network.optimizer.load_state_dict(checkpoint['target_optimizer'])
d3qn_agent.memory.memory=checkpoint['memory']
d3qn_agent.memory.prob=checkpoint['prob']
d3qn_agent.memory.lr=checkpoint['lr']
d3qn_agent.memory.counter=checkpoint['counter']

# Make environment
# Set 'record_action' to True because we need it to print results
env = rlcard.make('no-limit-holdem', config={'record_action': True})
human_agent = HumanAgent(env.num_actions)
#cfr_agent = models.load('no-limit-holdem-cfr').agents[0]
#d3qn_agent = D3QNAgent()
env.set_agents([human_agent, d3qn_agent])

print(">> No Limit Hold'em pre-trained model")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=True)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     D3QN Agent    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")