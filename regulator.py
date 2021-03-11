from agent import Agent
import numpy as np
import copy
from tabulate import tabulate
from matplotlib import pyplot as plt
from operator import itemgetter

class Policy(List):
    def __init__(self):
        self.policy = []
        return

    def add_level(self,new_level):
        assert type(new_level) == list , ('level input must be a list not:', type(new_level))
        assert len(new_level) == 3, ('level input must be length 3 not:', len(new_level))

        sel.policy.append(new_level)

        sorted(self.policy, key=itemgetter(0))

        for policy_level in self.policy:
            print ('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1], \
                   '--Levy rate', policy_level[2])
        return




class Regulator(Agent):

    def __init__(self, name, sim_time, notice_period):
        super().__init__(name, sim_time)
        assert type(notice_period) == int, 'notice period must be an integer'

        self.notice = notice_period
        self.level = 0
        self.changing = False
        self.time_to_change = 0
        self.emissions = np.float64(0)
        # arbitrary value for now
        self.limit = np.float64(100)

        return

    def set_emissions(self, new_emissions):
        assert type(new_emissions) == np.float64, ('emission input must be a float not:', type(new_emissions))
        self.emissions = new_emissions
        return

    def calculate_limit(self):

