from agent import Agent
import numpy as np
import math

# import copy
# from tabulate import tabulate
# from matplotlib import pyplot as plt

# from operator import itemgetter

# class Policy(list):
#     def __init__(self):
#         self.policy = []
#         return
#
#     def add_level(self, new_level):
#         assert type(new_level) == list, ('level input must be a list not:', type(new_level))
#         assert len(new_level) == 3, ('level input must be length 3 not:', len(new_level))
#
#         self.policy.append(new_level)
#
#         sorted(self.policy, key=itemgetter(0))
#
#         for policy_level in self.policy:
#             print('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate',
#             policy_level[1],
#                   '--Levy rate', policy_level[2])
#         return
#
#     def remove_level(self, ind):
#         assert type(ind) == int, ('should be an integer input not:', type(ind))
#         assert ind < len(self.policy), ('deleted level should exist. There are only', len(self.policy), 'levels. Not',
#                                         ind)
#         del self.policy[ind]
#         print('Successfully deleted level', ind)
#
#         for policy_level in self.policy:
#             print('Level:', self.policy.index(policy_level), '--Limit', policy_level[0], '--Tax rate',
#             policy_level[1],
#                   '--Levy rate', policy_level[2])
#         return
#
#     def level(self, lev):
#         assert type(lev) == int, ('should be an integer input not:', type(int))
#         assert lev < len(self.policy), ('level', lev, 'does not exist')
#
#         lev_copy = self.policy[lev][:]
#         # makes a copy of the level requested
#         return lev_copy


"""
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
        return lev_copy """


class Regulator(Agent):
    """def __init__(self, name, sim_time, env, notice_period, pol):
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
        print(' ------------- \n')
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
        return"""

    def __init__(self, name, sim_time, env, tax_rate, fraction, ratio_jump, start_levy, compliance_threshold=0.1,
                 notice_period=18, decade_jump=0.1):
        super().__init__(name, sim_time, env)
        assert type(notice_period) == int, 'notice period must be an integer'
        assert type(fraction) == float and 0.0 < fraction < 1.0, ("fraction input", fraction, 'must be a float '
                                                                                              'between 0 and 1')
        assert type(tax_rate) == float and 0.0 < tax_rate < 1.0, ("Starter tax rate input", tax_rate, 'must be a float '
                                                                                                      'between 0 and 1')
        assert type(decade_jump) == float and 0.0 < decade_jump, ("decade_jump input", decade_jump,
                                                                        'must be a float above 0')
        assert type(start_levy) == float, ("starting levy must be a float, not a", type(start_levy))
        assert type(ratio_jump) == float, (
        "ratio jump between level 0 and level 1 must be a float, not a", type(ratio_jump))
        assert type(compliance_threshold) == float and 0.0 < compliance_threshold < 1.0, (
            "compliance threshold input", compliance_threshold, "must be a float between 0 and 1")

        self.month = 0

        self.notice = notice_period
        self.level = 0
        self.fraction = fraction

        self.timer_exC = 0
        self.timer_decade = 0
        self.timer_punish = 0

        self.emissions = np.float64(0.0)
        self.c0 = np.float64(0.0)  # this will change once a function to make the history begins
        self.emissions_hist = []

        self.punish = 0
        self.punish_switch = False
        # This is a switch for compliance checking and punishment

        self.intercept = start_levy
        self.b = (ratio_jump * start_levy) * (3 / 10)
        self.a = self.b * (7 / 3)
        self.dec_jump = decade_jump
        self.comp_threshold = compliance_threshold
        # may add this variable into the initialisation

        self.changing_punish = False
        self.changing_decade = False
        self.changing_excess = bool(False)

        self.comp_check = False
        self.comp_timer = 0
        self.tax_rate = np.float64(tax_rate)
        self.levy_rate = np.float64(start_levy)
        self.future_levy_rate = np.float64(start_levy)

        return

    def set_emissions(self, new_emissions):
        assert type(new_emissions) == np.float64, ('emission input must be a float not:', type(new_emissions))
        self.emissions = new_emissions

        if len(self.emissions_hist) == 0:
            self.c0 = new_emissions

        self.emissions_hist.append(new_emissions)
        return

    def calculate_levy(self, intercept, level):
        assert type(intercept) == float, ("intercept input must be a float, not:", type(intercept))
        assert type(level) == int, ("level input must be an integer, not:", type(intercept))

        val = intercept + self.a * level + self.b * math.pow(level, 2)
        # the function may have to be changed to make the levy rates more significant
        return val

    def calculate_carbon(self, carbon):
        if self.punish_switch:
            return (carbon - self.c0) / (self.fraction * self.c0) + self.punish
        else:
            return (carbon - self.c0) / (self.fraction * self.c0)

    # fraction is the gap in emissions between levels. The gap is a fraction of the starter level.

    def asses_carbon_level(self):
        if self.changing_punish:
            return
        curr_carbon = self.calculate_carbon(self.emissions)
        if curr_carbon > (self.level + 1):
            self.trigger_exc_change()

        return

    def trigger_exc_change(self):
        self.changing_excess = True
        self.comp_check = True
        self.comp_timer = self.notice + 12
        self.exc_level_raise()
        return

    def exc_level_raise(self):
        self.timer_exC = self.notice
        new_levy = self.calculate_levy(self.intercept, (self.level + 1))
        self.future_levy_rate = np.float64(new_levy)
        return

    def comp_level_raise(self):
        if not self.punish_switch:
            return
        self.timer_punish = self.notice
        new_levy = self.calculate_levy(self.intercept, (self.level + 1))
        self.future_levy_rate = np.float64(new_levy)
        return

    def decade_level_change(self):
        self.timer_decade = 24  # this is so the decade change comes in two years from now
        new_intercept = (1 + self.dec_jump) * self.intercept  # the intercept will get raised pushing the levy curve up
        new_levy = self.calculate_levy(new_intercept, self.level)
        self.future_levy_rate = np.float64(new_levy)
        return

    def change_level_ex(self):
        self.level += 1
        print("Month", self.month, ", level", self.level, " Excessive emissions in month:", (self.month - self.notice))
        self.changing_excess = False
        return

    def change_level_punish(self):
        if not self.punish_switch:
            return
        self.punish += 1
        # so that subsequent level calculations take this punishment into account
        self.level += 1
        print("Month", self.month, ", level", self.level, " Failure to comply by:", (self.month - self.notice))

        self.changing_punish = False
        return

    def change_level_decade(self):
        self.intercept *= 1 + self.dec_jump
        #     now the next levy calculation will be increased
        self.changing_decade = False
        return

    def compliance_check(self):
        if not self.punish_switch:
            return
        if self.comp_check:
            self.comp_timer -= 1
            if self.comp_timer == 0:
                self.comp_check = False
                comp_level = self.emissions / self.emissions_hist[-(self.notice + 12)]
                if comp_level > self.comp_threshold:
                    self.changing_punish = True
                    self.comp_level_raise()

        return

    def decade_check(self):
        if self.month % 12 == 0:
            # checks if it's at the end of a year
            year = self.month / 12
            if (year + 2) % 10 == 0:
                # checks if it is 2 years before a new decade
                if self.changing_excess:
                    self.changing_excess = False
                    self.timer_exC = 0
                if self.changing_punish:
                    self.changing_punish = False
                    self.timer_punish = 0
                #     this sets the other levy changes to inactive so that the decade one dominates
                self.changing_decade = True
                self.comp_check = True
                self.comp_timer = 36
                self.decade_level_change()

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

    def time_to_change(self):
        if self.change_check():

            if self.changing_excess:
                return self.timer_exC

            elif self.changing_punish:
                return self.timer_punish

            elif self.changing_decade:
                return self.timer_decade
        else:
            return None

    def decrement_timer(self):
        if self.change_check():
            if self.changing_excess:
                self.timer_exC -= 1
                if self.timer_exC == 0:
                    self.change_level_ex()

            elif self.changing_punish:
                self.timer_punish -= 1
                if self.timer_punish == 0:
                    self.change_level_punish()

            elif self.changing_decade:
                self.timer_decade -= 1
                if self.timer_decade == 0:
                    self.change_level_decade()

        return

    def generate_levy_rate(self):
        self.levy_rate = np.float64(self.calculate_levy(self.intercept, self.level))
        # final function to change the levy rate
        return

    def iterate_regulator(self):
        self.set_emissions(self.env.aggregate['emissions'].value)
        # needs the clock function to take time from the simulator for processing too
        self.decade_check()
        if self.change_check():
            self.decrement_timer()
        else:
            self.compliance_check()
            self.asses_carbon_level()
        self.generate_levy_rate()
        return

    """To run this regulator
    make a policy table by adding in levels in the format [threshold, tax rate, levy rate]
    make the regulator with inputs notice period and the policy table
    each iteration
    give the regulator an emission stat and run. A tax-rate and levy_rate will be given"""
