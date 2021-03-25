from agent import Agent
import numpy as np
# import copy
# from tabulate import tabulate
# from matplotlib import pyplot as plt
import math
from operator import itemgetter


class Policy(List):
    def __init__(self):
        self.policy = []
        return

    def add_level(self, new_level):
        assert type(new_level) == list, ('level input must be a list not:', type(new_level))
        assert len(new_level) == 3, ('level input must be length 3 not:', len(new_level))

        self.policy.append(new_level)

        sorted(self.policy, key=itemgetter(0))

        for policy_level in self.policy:
            print('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1],
                  '--Levy rate', policy_level[2])
        return

    def remove_level(self, ind):
        assert type(ind) == int, ('should be an integer input not:', type(ind))
        assert ind < len(self.policy), ('deleted level should exist. There are only', len(self.policy), 'levels. Not',
                                        ind)
        del self.policy[ind]
        print('Successfully deleted level', ind)

        for policy_level in self.policy:
            print('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1],
                  '--Levy rate', policy_level[2])
        return

    def level(self, lev):
        assert type(lev) == int, ('should be an integer input not:', type(int))
        assert lev < len(self.policy), ('level', lev, 'does not exist')

        lev_copy = self.policy[lev][:]
        # makes a copy of the level requested
        return lev_copy


class Regulator(Agent):
    """def __init__(self, name, sim_time, notice_period, pol):
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
        self.tax_rate = np.float64(0.19)
        self.levy_rate = np.float64(5)

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
        # the number 5 is arbitrary. Haven't picked a value to multiply emissions by for the damage calc. may end up being 1
        if e_damage > self.limit:
            self.punish()

        return

    def punish(self):
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
            self.changing = False
        return

    def calc_tax_rate(self):
        tax_rate = self.pol_table.level(self.level)[1]
        self.tax_rate = tax_rate
        return

    def calc_levy_rate(self):
        levy = self.pol_table.level(self.level)[2]
        self.levy_rate = levy
        return

    def iterate_Regulator(self, emission_rate):
        self.set_emissions()
        self.compute_limit()
        self.calc_environmental_damage()
        self.retrieve_level()
        self.calc_tax_rate()
        self.calc_levy_rate()
        return"""

    def __init__(self, name, sim_time, notice_period, fraction, start_levy):
        super().__init__(name, sim_time)
        assert type(notice_period) == int, 'notice period must be an integer'
        assert type(fraction) == float and 0.0 < fraction < 1.0, ("fraction input", fraction, "must be a float "
                                                                                              "between 0 and 1")
        assert type(start_levy) == float, ("starting levy must be a float, not a", type(start_levy))

        self.notice = notice_period
        self.level = 0
        self.fraction = fraction

        self.timer_exC = 0
        self.timer_decade = 0
        self.timer_punish = 0

        self.emissions = np.float64(0)
        self.c0 = np.float64(0.0)  # this will change once a function to make the history begins
        self.emissions_hist = []

        self.punish = 0
        self.intercept = start_levy

        self.changing_punish = False
        self.changing_decade = False
        self.changing_excess = bool(False)

        self.comp_check = False
        self.comp_timer = 0
        # arbitrary value for now
        # self.limit = np.float64(100)
        # self.pol_table = pol
        self.tax_rate = np.float64(0.19)
        self.levy_rate = np.float64(5)

        return

    def set_emissions(self, new_emissions):
        assert type(new_emissions) == np.float64, ('emission input must be a float not:', type(new_emissions))
        self.emissions = new_emissions
        self.emissions_hist.append(new_emissions)
        return

    def calculate_levy(self, intercept, level):
        assert type(intercept) == float, ("intercept input must be a float, not:", type(intercept))
        val = intercept + 0.1 * level + 0.04 * math.pow(level, 2)
        # the function may have to be changed to make the levy rates more significant
        return val

    def calculate_Carbon(self, carbon, fraction, c0):
        return (carbon - c0) / (fraction * c0) + self.punish

    def asses_carbon_level(self):
        curr_carbon = self.calculate_Carbon(self.emissions, self.fraction, self.c0)
        if curr_carbon > self.level:
            self.trigger_exC_change()

        return

    def trigger_exC_change(self):
        self.changing_excess = True
        self.comp_check = True
        self.comp_timer = self.notice + 12
        self.exC_level_raise()
        return

    def exC_level_raise(self):
        self.exC_timer = self.notice
        new_levy = self.calculate_levy(self.intercept, (self.level + 1))
        # now this new levy needs to be broadcasted to the system
        return

    def change_check(self):
        if self.changing_excess:
            return True
        elif self.changing_punish:
            return True
        elif self.changing_decade:
            return True
        else:
            return False

    """To run this regulator
    make a policy table by adding in levels in the format [threshold, tax rate, levy rate]
    make the regulator with inputs notice period and the policy table
    each iteration
    give the regulator an emmision stat and run. A tax-rate and levy_rate will be given"""
