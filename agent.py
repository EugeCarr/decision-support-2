"""This file defines the Agent class, and subclasses thereof, for the agent-based simulation"""
import numpy as np
from parameter import Parameter
import parameter as par


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


class Agent(object):
    def __init__(self, name, sim_time):
        assert type(name) == str, ('name must be a string. input value is a', type(name))
        assert type(sim_time) == int, ('sim_time must be an integer. input value is a', type(sim_time))
        assert sim_time > 0, 'sim_time must be greater than zero'
        self.name = name
        self.month = int(0)

        print(' ================================ \n', self.name, 'created \n ================================')
        return


class Manufacturer(Agent):
    # object initialisation
    def __init__(self, name, sim_time, initial_production_volume=np.float64(1000)):
        super().__init__(name, sim_time)
        """To add a new parameter, define it as an attribute of object type Parameter. Then add it to the correct list
        of variables (dependent or independent) in the correct place so parameters are computed in the right order. 
        The argument fun is the function which, given an argument of object type Agent, returns a single float which
        is the next value of the parameter. 
        The argument project is a function which, given an argument of object type Agent, returns an array of size n
        which is the projected value of the parameter for the next n months.
        Optional argument init is the initial value of the parameter. This is usually unnecessary.
        history_time and projection_time should only be changed from their default values with great care."""

        self.projection_time = 120  # how many months into the future will be predicted?

        # define independent variables for current time
        self.production_volume = Parameter(par.production_volume, par.production_volume_projection, sim_time,
                                           init=initial_production_volume)
        # total PET production per annum, starts at 1000
        self.unit_sale_price = Parameter(par.unit_sale_price, par.unit_sale_price_projection, sim_time)
        # sale price of one unit of PET
        self.unit_feedstock_cost = Parameter(par.unit_feedstock_cost, par.unit_feedstock_cost_projection, sim_time)
        # feedstock cost per unit of PET produced
        self.unit_process_cost = Parameter(par.unit_process_cost, par.unit_process_cost_projection, sim_time)
        # cost of running process per unit of PET produced

        self.bio_feedstock_cost = Parameter(par.bio_feedstock_cost, par.bio_feedstock_cost_projection, sim_time)
        # bio feedstock cost per unit of PET produced
        self.bio_process_cost = Parameter(par.bio_process_cost, par.bio_process_cost_projection, sim_time)
        # cost of process per unit of PET from bio routes
        self.proportion_bio = Parameter(par.proportion_bio, par.proportion_bio_projection, sim_time)
        # proportion of production from biological feedstocks
        self.levy_rate = Parameter(par.levy_rate, par.levy_rate_projection, sim_time, init=np.float64(0.2))

        self.bio_capacity = Parameter(par.bio_capacity, par.bio_capacity_projection, sim_time)
        self.fossil_capacity = Parameter(par.fossil_capacity, par.fossil_capacity_projection, sim_time,
                                         init=initial_production_volume)

        self.expansion_cost = Parameter(par.expansion_cost, par.expansion_cost_projection, sim_time)
        # cost of increasing production capacity
        self.gross_profit = Parameter(par.gross_profit, par.gross_profit_projection, sim_time)
        # profits after levies and before taxes
        self.emissions = Parameter(par.emissions, par.emissions_projection, sim_time)
        # emissions from manufacturing PET from fossil fuels
        self.tax_payable = Parameter(par.tax_payable, par.tax_payable_projection, sim_time)
        self.levies_payable = Parameter(par.levies_payable, par.levies_payable_projection, sim_time)
        self.net_profit = Parameter(par.net_profit, par.net_profit_projection, sim_time)
        # monthly profit after tax and levies
        self.profitability = Parameter(par.profitability, par.profitability_projection, sim_time)
        # profitability (net profit per unit production)
        self.liquidity = Parameter(par.liquidity, par.liquidity_projection, sim_time, init=np.float64(5000))
        # accumulated cash
        self.profit_margin = Parameter(par.profit_margin, par.profit_margin_projection, sim_time)

        # dictionary of all variables in the order in which they should be computed
        self.parameter = {
            'production_volume': self.production_volume,
            'unit_sale_price': self.unit_sale_price,
            'unit_feedstock_cost': self.unit_feedstock_cost,
            'unit_process_cost': self.unit_process_cost,
            'bio_feedstock_cost': self.bio_feedstock_cost,
            'bio_process_cost': self.bio_process_cost,
            'proportion_bio': self.proportion_bio,
            'levy_rate': self.levy_rate,
            'bio_capacity': self.bio_capacity,
            'fossil_capacity': self.fossil_capacity,
            'expansion_cost': self.expansion_cost,
            'emissions': self.emissions,
            'levies_payable': self.levies_payable,
            'gross_profit': self.gross_profit,
            'tax_payable': self.tax_payable,
            'net_profit': self.net_profit,
            'profitability': self.profitability,
            'liquidity': self.liquidity,
            'profit_margin': self.profit_margin
        }

        # list of keys in the dictionary in the order passed to the object on initialisation
        # ensures computation order is preserved
        self.keys = list(self.parameter.keys())

        # now define other parameters which will not be recorded or projected
        self.proportion_bio_target = np.float64()  # target value for the proportion of production via bio route
        self.projection_met = False  # 1 or 0 depending on whether the next target will be met by current
        # projection

        self.tax_rate = np.float64(0.19)  # current tax on profits, starts at 19%
        self.emissions_rate = np.float64(5)  # units of emissions per unit of PET produced from non-bio route

        self.levy_rate_changing = False
        self.time_to_levy_change = int()
        self.future_levy_rate = np.float64()

        # additional projection variables
        self.tax_rate_projection = np.ones(self.projection_time) * self.tax_rate

        # define variables for the targets against which projections are measured
        # and the times at which they happen
        self.target_value = np.float64(0.35)  # currently fixed values
        self.target1_year = 5
        self.target2_year = 10

        self.beyond_target_range = False  # a boolean set to true if the simulation runs beyond the point for which
        # targets are defined

        self.proportion_change_rate = np.float64(0.1 / 9)  # greatest possible monthly change in self.proportion_bio
        self.implementation_delay = int(15)  # time delay between investment decision and movement of bio_proportion
        self.implementation_countdown = int(0)  # countdown to start of increase in bio proportion
        self.under_construction = False  # is change in bio capacity occurring?

        self.fossil_capacity_cost = np.float64(10)  # one-time cost of increasing production capacity for fossil route
        self.bio_capacity_cost = np.float64(12)  # one-time cost of increasing production capacity for bio route

        # output initialisation state to console
        print(' INITIAL STATE \n -------------'
              '\n Annual production volume:', self.production_volume.value,
              '\n Projection horizon (months):', self.projection_time,
              '\n Target profitability:', self.target_value,
              '\n -------------')

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
            if self.profitability.projection[time_to_target1] >= self.target_value:
                self.projection_met = True
            else:
                self.projection_met = False

        else:
            if self.profitability.projection[time_to_target2] >= self.target_value:
                self.projection_met = True
            else:
                self.projection_met = False

        return

    def investment_decision(self):
        # decision logic for increasing investment in biological process route
        if self.projection_met:
            pass
        elif not self.projection_met:
            while self.proportion_bio_target < 1 and not self.projection_met:

                if not self.under_construction:
                    self.implementation_countdown = self.implementation_delay

                self.proportion_bio_target += 0.05
                if self.proportion_bio_target > 1:
                    self.proportion_bio_target = 1

                self.project_variables()
                self.projection_check()

                if self.proportion_bio_target == 1 and not self.projection_met:
                    print('Month:', self.month, '\n next profitability target could not be met'
                                                'at any bio proportion target')

            else:
                pass

        else:
            pass

        return

    def time_step(self):
        if self.implementation_countdown > 0:
            self.implementation_countdown -= 1

        self.update_variables()
        if self.month % 12 == 1:
            self.project_variables()
            self.projection_check()
            self.investment_decision()
        self.record_timestep()
        return
