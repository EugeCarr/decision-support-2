from agent import Agent
import numpy as np
from operator import itemgetter


class Policy(list):
    def __init__(self):
        super().__init__()
        return

    def add_level(self, new_level, print_new_policy=False):
        assert type(new_level) == list, ('level input must be a list not:', type(new_level))
        assert len(new_level) == 3, ('level input must be length 3 not:', len(new_level))

        self.append(new_level)

        sorted(self, key=itemgetter(0))

        print(' Policy updated')
        if print_new_policy:
            print('New policy levels are:')
            for policy_level in self:
                print('Level:', self.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1],
                      '--Levy rate', policy_level[2])
        return

    def remove_level(self, ind, print_new_policy=False):
        assert type(ind) == int, ('should be an integer input not:', type(ind))
        assert ind < len(self), ('deleted level should exist. There are only', len(self), 'levels. Not',
                                        ind)
        del self[ind]
        print('Successfully deleted level', ind)

        if print_new_policy:
            print('New policy levels are:')
            for policy_level in self:
                print('Level:', self.index(policy_level), '--Limit', policy_level[0], '--Tax rate', policy_level[1],
                      '--Levy rate', policy_level[2])
        return

    def level(self, lev):
        assert type(lev) == int, ('should be an integer input not:', type(lev))
        assert lev < len(self), ('level', lev, 'does not exist')

        lev_copy = self[lev][:]
        # makes a copy of the level requested
        return lev_copy


class Regulator(Agent):

    def __init__(self, name, sim_time, env, notice_period, pol):
        super().__init__(name, sim_time, env)
        assert type(notice_period) == int, 'notice period must be an integer'
        assert isinstance(pol, Policy)
        self.notice = notice_period
        self.level = 0
        self.changing = bool(False)
        self.time_to_change = 0
        self.emissions = np.float64()
        # arbitrary value for now
        self.limit = np.float64()
        self.pol_table = pol
        self.tax_rate = np.float64()
        self.levy_rate = np.float64()

        self.max_level_reached = False

        print(' POLICY DEFINITION \n -------------')
        for policy_level in self.pol_table:
            print(' Level:', self.pol_table.index(policy_level), '--Limit', policy_level[0],
                  '--Tax rate', policy_level[1], '--Levy rate', policy_level[2])
        print(' -------------')
        return

    def set_emissions(self, new_emissions):
        assert type(new_emissions) == np.float64, ('emission input must be a float not:', type(new_emissions))
        self.emissions = new_emissions
        return

    def compute_limit(self):
        if self.level < len(self.pol_table) - 1:
            self.limit = self.pol_table.level(self.level + 1)[0]
        # if there is another level above, go to that
        # else if this is the first time reaching the maximum set the limit to infinity
        elif not self.max_level_reached:
            self.limit = np.inf
            print('Month', self.month, '- Highest level of emissions regulation reached:', self.level, self.levy_rate)
            self.max_level_reached = True
        else:
            pass

        # gets the list of limit, tax, levy from the policy table

        return

    def calc_environmental_damage(self):
        e_damage = 5 * self.emissions
        # the number 5 is arbitrary. Haven't picked a value to multiply emissions by for the damage calc. may end up
        # being 1
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

    def iterate_regulator(self, emission_rate):
        self.set_emissions(emission_rate)
        self.compute_limit()
        self.calc_environmental_damage()
        self.retrieve_level()
        self.calc_tax_rate()
        self.calc_levy_rate()
        return

    """To run this regulator
    make a policy table by adding in levels in the format [threshold, tax rate, levy rate]
    make the regulator with inputs notice period and the policy table
    each iteration
    give the regulator an emission stat and run. A tax-rate and levy_rate will be given"""
