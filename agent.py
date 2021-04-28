"""This file defines the Agent class, and subclasses thereof, for the agent-based simulation"""
import numpy as np
import parameter as par
import copy
from scipy import optimize
from scipy.optimize import Bounds


def run_check():
    proceed = str()
    while True:
        try:
            proceed = str(input('Do you want to proceed? Y/N:'))
        except ValueError:
            continue

        if proceed.lower() not in 'yn':
            continue
        else:
            break

    if proceed.lower() == 'n':
        raise SystemExit(0)
    elif proceed.lower() == 'y':
        pass


class Environment(object):
    def __init__(self, variables, aggregates):
        assert type(variables) == dict
        assert type(aggregates) == dict
        for value in list(variables.values()):
            assert isinstance(value, par.Environment_Variable)
        for value in list(aggregates.values()):
            assert isinstance(value, par.Environment_Variable)

        self.month = int(0)

        self.parameter = variables
        self.aggregate = aggregates

        self.levy_rate_changing = False
        self.time_to_levy_change = int()
        self.future_levy_rate = np.float64()
        return

    def reset_aggregates(self):
        for item in list(self.aggregate.keys()):
            self.aggregate[item].value = 0
        return


class Agent(object):
    def __init__(self, name, sim_time, env):
        assert type(name) == str, ('name must be a string. input value is a', type(name))
        assert type(sim_time) == int, ('sim_time must be an integer. input value is a', type(sim_time))
        assert sim_time > 0, 'sim_time must be greater than zero'
        assert type(env) == Environment

        self.name = name
        self.month = int(0)
        self.env = env
        self.sim_time = sim_time

        print('\n ================================ \n', self.name, 'created \n ================================')
        return


def utility_func(manufacturer):
    assert isinstance(manufacturer, Manufacturer)

    time_to_target = 60

    under = (manufacturer.target_value -
             manufacturer.parameter[manufacturer.value_function].projection[time_to_target])

    if under < 0:
        manufacturer.projection_met = True
    else:
        manufacturer.projection_met = False

    return under


class Manufacturer(Agent):
    # object initialisation
    def __init__(self, name, sim_time, env, parameters, value_function='profitability', target_value=0.35,
                 capacity_root_coefficient =2.0):
        super().__init__(name, sim_time, env)
        """To add a new parameter, define it in the dictionary as a Parameter object in the correct place so that 
        parameters are computed in the right order."""
        assert type(value_function) == str
        assert value_function in parameters
        assert type(capacity_root_coefficient) == float
        for value in list(parameters.values()):
            assert isinstance(value, par.Parameter)

        self.value_function = value_function
        self.target_value = np.float64(target_value)

        self.projection_time = 120  # how many months into the future will be predicted?

        # dictionary of all variables in the order in which they should be computed
        self.parameter = parameters

        self.capacity_root_coefficient = capacity_root_coefficient
        # this is the value that is used to determine the function for capacity expansions

        # list of keys in the dictionary in the order passed to the object on initialisation
        # ensures computation order is preserved
        self.keys = list(self.parameter.keys())

        # now define other parameters which will not be recorded or projected
        self.proportion_bio_target = np.float64(0)  # target value for the proportion of production via bio route
        self.projection_met = False  # boolean dependent on whether the next target will be met by current
        # projection

        self.tax_rate = np.float64(0.19)  # tax on profits, 19%
        self.emissions_rate = np.float64(5)  # units of emissions per unit of PET produced from non-bio route

        # additional projection variables
        self.tax_rate_projection = np.ones(self.projection_time) * self.tax_rate

        self.proportion_change_rate = np.float64(0.1 / 9)  # greatest possible monthly change in self.proportion_bio
        self.implementation_delay = int(15)  # time delay between investment decision and movement of bio_proportion
        self.implementation_countdown = int(0)  # countdown to start of increase in bio proportion
        self.under_construction = False  # is change in bio capacity occurring?

        self.fossil_capacity_target = np.float64(1000)
        self.bio_capacity_target = np.float64(0)

        self.change_rate = 100  # maximum amount of production capacity that can be built/decommissioned in a month
        self.design_time = int(15)  # delay between decision to build and start of construction if not already building

        self.fossil_build_countdown = int(0)
        self.fossil_building = False
        self.bio_build_countdown = int(0)
        self.bio_building = False
        self.bio_building_month = int(0)

        self.fossil_capacity_cost = np.float64(10)  # capital cost of 1 unit/yr production capacity for fossil route
        self.bio_capacity_cost = np.float64(12)  # capital cost of 1 unit/yr production capacity for bio route

        self.fossil_resource_ratio = np.float64(1)  # no. of units of fossil resource used per unit of PET produced
        self.bio_resource_ratio = np.float64(1)  # no. of units of bio resource used per unit of PET produced

        self.capacity_maintenance_cost = np.float64(0.001)  # cost of maintaining manufacturing
        # capacity per unit per month

        self.negative_liquidity = False

        self.fossil_utilisation_target = 0.9  # capacity utilisation targets
        self.bio_utilisation_target = 0.9

        # output initialisation state to console
        print(' INITIAL STATE \n -------------'
              '\n Annual production volume:', self.parameter['total_production'].value,
              '\n Projection horizon (months):', self.projection_time,
              '\n Target profitability:', self.target_value,
              '\n ------------- \n')

        # run_check()

        return

    # region -- methods for updating values and projections
    def update_variables(self):
        for key in self.keys:
            self.parameter[key].update(self)
        return

    def record_timestep(self):
        # method to write current variables (independent and dependent) to records
        for key in self.keys:
            self.parameter[key].record(self.month)
        return

    def project_variables(self):
        # calculate projections for all variables
        for key in self.keys:
            self.parameter[key].forecast(self)
        return

    # endregion

    def projection_check(self):
        # checks whether the profitability target will be met on the basis of latest projection
        # at the next 5-year interval
        # when the next 5-year interval is less than 1 year away the logic is based on the following interval

        time_to_target1 = 60 - self.month % 60
        time_to_target2 = time_to_target1 + 60

        if time_to_target1 > 12:
            if self.parameter[self.value_function].projection[time_to_target1] >= self.target_value:
                self.projection_met = True
            else:
                self.projection_met = False

        else:
            if self.parameter[self.value_function].projection[time_to_target2] >= self.target_value:
                self.projection_met = True
            else:
                self.projection_met = False

        return

    # def investment_decision(self):
    #     # decision logic for increasing investment in biological process route
    #     if self.projection_met:
    #         pass
    #     else:
    #         while self.proportion_bio_target < 1 and not self.projection_met:
    #
    #             if not self.under_construction:
    #                 self.implementation_countdown = self.implementation_delay
    #
    #             self.proportion_bio_target += 0.05
    #             if self.proportion_bio_target > 1:
    #                 self.proportion_bio_target = 1
    #
    #             self.project_variables()
    #             self.projection_check()
    #
    #             if self.proportion_bio_target == 1 and not self.projection_met:
    #                 print('Month:', self.month, '\n next profitability target could not be met'
    #                                             'at any bio proportion target')
    #
    #     return

    def capacity_scenario(self, targets):
        assert isinstance(targets, np.ndarray)
        assert len(targets) == 2
        fossil_target = targets[0]
        bio_target = targets[1]

        sandbox = copy.deepcopy(self)
        sandbox.fossil_capacity_target = fossil_target
        sandbox.bio_capacity_target = bio_target

        sandbox.project_variables()

        utility = utility_func(sandbox)

        return utility

    def optimal_strategy(self):
        current_fossil = self.parameter['fossil_capacity'].value
        current_bio = self.parameter['bio_capacity'].value

        max_fossil = self.parameter['fossil_capacity_max'].value
        max_bio = self.parameter['bio_capacity_max'].value

        x0 = np.array([current_fossil, current_bio])

        res = optimize.minimize(self.capacity_scenario, x0,
                                method='l-bfgs-b', bounds=Bounds([0.0, 0.0], [max_fossil, max_bio]))

        targets = res.x

        return targets

    def time_step_alt(self):
        if self.fossil_build_countdown > 0:
            self.fossil_build_countdown -= 1
            if self.fossil_build_countdown == 0:
                self.fossil_building = True
        if self.bio_build_countdown > 0:
            self.bio_build_countdown -= 1
            if self.bio_build_countdown == 0:
                self.bio_build_countdown = True

        self.update_variables()
        if self.month % 12 == 1:
            self.project_variables()
            self.projection_check()

            if not self.projection_met:
                new_targets = self.optimal_strategy()
                self.fossil_capacity_target = new_targets[0]
                self.bio_capacity_target = new_targets[1]

                if not self.bio_building:
                    self.bio_build_countdown = self.design_time
                if not self.fossil_building:
                    self.fossil_build_countdown = self.design_time

                self.project_variables()
                self.projection_check()
        self.record_timestep()
        return

    # def time_step(self):
    #     if self.implementation_countdown > 0:
    #         self.implementation_countdown -= 1
    #
    #     self.update_variables()
    #     if self.month % 12 == 1:
    #         self.project_variables()
    #         self.projection_check()
    #         self.investment_decision()
    #     self.record_timestep()
    #     return
