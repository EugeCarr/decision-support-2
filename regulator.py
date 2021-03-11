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

    def add_level(self, new_level):
        assert type(new_level) == list, ('level input must be a list not:', type(new_level))
        assert len(new_level) == 3, ('level input must be length 3 not:', len(new_level))

        sel.policy.append(new_level)

        sorted(self.policy, key=itemgetter(0))

        for policy_level in self.policy:
            print('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1], \
                  '--Levy rate', policy_level[2])
        return

    def remove_level(self, ind):
        assert type(ind) == int, ('should be an integer input not:', type(ind))
        assert ind < len(self.policy), ('deleted level should exist. There are only', len(self.policy), 'levels. Not', \
                                        ind)
        del self.policy[ind]
        print('Successfully deleted level', ind)

        for policy_level in self.policy:
            print('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1], \
                  '--Levy rate', policy_level[2])
        return

    def level(self, lev):
        assert type(lev) == int, ('should be an integer input not:', type(ind))
        assert lev < len(self.policy), ('level', lev, 'does not exist')

        lev_copy = self.policy[lev][:]
        # makes a copy of the level requested
        return lev_copy


class Regulator(Agent):

    def __init__(self, name, sim_time, notice_period, pol):
        super().__init__(name, sim_time)
        assert type(notice_period) == int, 'notice period must be an integer'
        assert isinstance(pol, Policy)
        self.notice = notice_period
        self.level = 0
        self.changing = bool(False)
        self.time_to_change = 0
        self.emissions = np.float64(0)
        # arbitrary value for now
        self.limit = np.float64(100)
        self.pol_table = pol

        return

    def set_emissions(self, new_emissions):
        assert type(new_emissions) == np.float64, ('emission input must be a float not:', type(new_emissions))
        self.emissions = new_emissions
        return

    def compute_limit(self):
        next_limit = self.pol_table.level(self.level + 1)
        # gets the list of limit, tax, levy from the policy table
        self.limit = next_limit[0]
        return

    def calc_environmental_damage(self):
        e_damage = 5 * self.emissions
        if e_damage > self.limit:
            self.decentivise()

        return

    def decentivise(self):
        if not self.changing:
            self.changing = True
            self.level_raise()

        return

    def level_raise(self):
        self.time_to_change = self.notice
        # the idea is to have a message sent out when this level is raised.
        # this will allow the PET manufacturer to update projections
        return

    def retrieve_level(self):
        if self.changing:
            self.time_to_change -= 1
            self.change_level()

        return

    def change_level(self):
        if self.time_to_change == 0:
            self.level += 1
        return

    def get_tax_rate(self):
        tax_rate = self.pol_table.level(self.level)[1]

        return tax_rate

    def get_levy_rate(self):
        levy = self.pol_table.level(self.level)[2]
        return levy
