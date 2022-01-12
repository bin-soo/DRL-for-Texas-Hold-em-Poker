import numpy as np

class BAgent(object):
    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.memory = Memory()
        self.count = 0
    
    @staticmethod
    def step(state):
        ''' Predict the action given the current state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        if self.count < 100000:
            probs = [0 for _ in range(self.num_actions)]
            for i in state['legal_actions']:
                probs[i] = 1/len(state['legal_actions'])

            info = {}
            info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}
            self.count += 1
            return np.random.choice(list(state['legal_actions'].keys())), info
        else:
            winrate = self.memory.wincount / (self.memory.wincount + self.memory.losecount)
            loserate = self.memory.losecount / (self.memory.wincount + self.memory.losecount)
            count = 0
            choice = 0
            for i in range(52):
                if state['obs'][i] == 1:
                    count = count + 1
                    winrate = winrate * self.memory.win[i] / self.memory.wincount
                    loserate = loserate * self.memory.lose[i] / self.memory.losecount
            wincom = winrate / (winrate + loserate)
            if count == 2:
                if wincom < 0.2:
                    choice = 0
                elif wincom < 0.6:
                    choice = 1
                elif wincom < 0.8:
                    choice = 2
                elif wincom < 0.95:
                    choice = 3
                else:
                    choice = 4
            elif count == 5:
                if wincom < 0.3:
                    choice = 0
                elif wincom < 0.6:
                    choice = 1
                elif wincom < 0.8:
                    choice = 2
                elif wincom < 0.9:
                    choice = 3
                else:
                    choice = 4
            elif count == 6:
                if wincom < 0.35:
                    choice = 0
                elif wincom < 0.6:
                    choice = 1
                elif wincom < 0.75:
                    choice = 2
                elif wincom < 0.85:
                    choice = 3
                else:
                    choice = 4
            elif count == 7:
                if wincom < 0.4:
                    choice = 0
                elif wincom < 0.6:
                    choice = 1
                elif wincom < 0.7:
                    choice = 2
                elif wincom < 0.8:
                    choice = 3
                else:
                    choice = 4
            if len(list(state['legal_actions'].keys())) == 2:
                if wincom >= 0.5:
                    choice = 1
                else:
                    choice = 0
            elif len(list(state['legal_actions'].keys())) == 3:
                if count == 2:
                    if wincom < 0.2:
                        choice = 0
                    elif wincom < 0.8:
                        choice = 1
                    else:
                        choice = 4
                elif count == 5:
                    if wincom < 0.3:
                        choice = 0
                    elif wincom < 0.8:
                        choice = 1
                    else:
                        choice = 4
                elif count == 6:
                    if wincom < 0.35:
                        choice = 0
                    elif wincom < 0.75:
                        choice = 1
                    else:
                        choice = 4
                elif count == 7:
                    if wincom < 0.4:
                        choice = 0
                    elif wincom < 0.7:
                        choice = 1
                    else:
                        choice = 4
            elif 2 not in list(state['legal_actions'].keys()):
                if count == 2:
                    if wincom < 0.2:
                        choice = 0
                    elif wincom < 0.7:
                        choice = 1
                    elif wincom < 0.95:
                        choice = 3
                    else:
                        choice = 4
                elif count == 5:
                    if wincom < 0.3:
                        choice = 0
                    elif wincom < 0.7:
                        choice = 1
                    elif wincom < 0.9:
                        choice = 3
                    else:
                        choice = 4
                elif count == 6:
                    if wincom < 0.35:
                        choice = 0
                    elif wincom < 0.65:
                        choice = 1
                    elif wincom < 0.85:
                        choice = 3
                    else:
                        choice = 4
                elif count == 7:
                    if wincom < 0.4:
                        choice = 0
                    elif wincom < 0.65:
                        choice = 1
                    elif wincom < 0.8:
                        choice = 3
                    else:
                        choice = 4
            elif 3 not in list(state['legal_actions'].keys()):
                if count == 2:
                    if wincom < 0.2:
                        choice = 0
                    elif wincom < 0.6:
                        choice = 1
                    elif wincom < 0.9:
                        choice = 2
                    elif wincom < 0.9:
                        choice = 4
                elif count == 5:
                    if wincom < 0.3:
                        choice = 0
                    elif wincom < 0.6:
                        choice = 1
                    elif wincom < 0.85:
                        choice = 2
                    else:
                        choice = 4
                elif count == 6:
                    if wincom < 0.35:
                        choice = 0
                    elif wincom < 0.6:
                        choice = 1
                    elif wincom < 0.8:
                        choice = 2
                    else:
                        choice = 4
                elif count == 7:
                    if wincom < 0.4:
                        choice = 0
                    elif wincom < 0.6:
                        choice = 1
                    elif wincom < 0.75:
                        choice = 2
                    else:
                        choice = 4
            self.count += 1
            return choice, choice


class Memory(object):
    def __init__(self):
        self.win = []
        self.lose = []
        for i in range(52):
            self.win.append(1)
            self.lose.append(1)
        self.wincount = 0
        self.losecount = 0

    def save(self, state, chips):
        if chips >= 0:
            self.wincount += 1
            for i in range(52):
                self.win[i] += state['obs'][i]
        else:
            self.losecount += 1
            for i in range(52):
                self.lose[i] += state['obs'][i]

    
