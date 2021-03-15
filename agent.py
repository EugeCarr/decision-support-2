"""This file defines the Agent class, and subclasses thereof, for the agent-based simulation"""
import numpy as np


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


class Parameter(object):
    def __init__(self, fun, history_time=120, projection_time=120, init=0):
        assert type(history_time) == int
        assert history_time > 0
        assert type(projection_time) == int
        assert projection_time > 0

        self.fun = fun

        self.value = np.float64(init)

        self.projection = np.zeros(projection_time)
        self.history = np.zeros(history_time)

        return

    def update(self, agent):
        assert isinstance(agent, Agent)
        self.value = self.fun(agent)
        return

    def record(self, time):
        self.history[time] = self.value
        return


def production_volume(agent) -> np.float64:
    volume = agent.production_volume.value
    month = agent.month

    # production volume is defined by growth rates in 2 periods

    sim_period_0 = 5  # end year for first simulation period
    sim_period_0_months = sim_period_0 * 12  # end month for first simulation period
    growth_rate_0 = 1.02  # YoY growth rate for the first simulation period, expressed as a ratio
    growth_rate_0_monthly = np.power(growth_rate_0, 1 / 12)  # annual growth rate changed to month-on-month

    sim_period_1 = 10  # end year for second simulation period
    sim_period_1_months = sim_period_1 * 12  # end month for second simulation period
    growth_rate_1 = 1.03  # YoY growth rate for the second simulation period, expressed as a ratio
    growth_rate_1_monthly = np.power(growth_rate_1, 1 / 12)  # annual growth rate changed to month-on-month

    if month <= sim_period_0_months:
        val = volume * growth_rate_0_monthly

    elif month <= sim_period_1_months:
        val = volume * growth_rate_1_monthly

    else:
        raise ValueError('production growth not defined for month', month)

    return val


def unit_sale_price(agent) -> np.float64:
    # unit sale price is given by a normal distribution
    mean = np.float64(6)
    std_dev = 0.01
    val = np.float64(np.random.normal(mean, std_dev, None))
    return val


def unit_feedstock_cost(agent):
    # unit feedstock cost is given by a normal distribution
    mean = np.float64(2)
    std_dev = 0.01
    val = np.random.normal(mean, std_dev, None)
    return val


def unit_process_cost(agent):
    # process cost is given by a normal distribution around a mean which is a weakly decreasing function of
    # production volume, such that a doubling in production reduces processing unit cost by 10%, starting from 1
    mean = 2.8576 / np.power(agent.production_volume.value, 0.152)
    std_dev = 0.005
    val = np.random.normal(mean, std_dev, None)
    return val


def bio_feedstock_cost(agent):
    # unit feedstock cost is given by a normal distribution
    mean = np.float64(2)
    std_dev = 0.01
    val = np.random.normal(mean, std_dev, None)
    return val


def bio_process_cost(agent):
    # process cost is given by a normal distribution
    mean = 1.05
    std_dev = 0.005
    val = np.random.normal(mean, std_dev, None)
    return val


def proportion_bio(agent):
    # monthly change in bio proportion is either the amount to reach the target value, or else the maximum change
    val = np.float64(agent.proportion_bio.value)
    if agent.implementation_countdown == 0 and agent.proportion_bio.value != agent.proportion_bio_target:
        agent.under_construction = True
        distance_from_target = agent.proportion_bio_target - agent.proportion_bio.value
        if abs(distance_from_target) < agent.proportion_change_rate:
            val = agent.proportion_bio_target
        elif distance_from_target > 0:
            val += agent.proportion_change_rate
        elif distance_from_target < 0:
            val -= agent.proportion_change_rate
        else:
            pass
    else:
        agent.under_construction = False

    return val


def emissions(agent):
    fossil_production = agent.production_volume.value / 12 * (1 - agent.proportion_bio.value)
    val = fossil_production * agent.emissions_rate
    return val


def levies_payable(agent):
    """This will calculate the levies payable on production/consumption/emission, once they are defined"""
    val = agent.levy_rate * agent.emissions.value
    return val


def gross_profit(agent):
    production_in_month = agent.production_volume.value / 12
    revenue = production_in_month * agent.unit_sale_price.value
    costs = (production_in_month * ((1 - agent.proportion_bio.value) *
                                    (agent.unit_feedstock_cost.value + agent.unit_process_cost.value) +
                                    agent.proportion_bio.value *
                                    (agent.bio_feedstock_cost.value + agent.bio_process_cost.value))
             + agent.levies_payable.value)
    val = revenue - costs
    return val


def tax_payable(agent):
    val = agent.gross_profit.value * agent.tax_rate
    return val


def net_profit(agent):
    val = agent.gross_profit.value - agent.tax_payable.value
    return val


def profitability(agent):
    val = agent.net_profit.value / (agent.production_volume.value / 12)
    return val


class Agent(object):
    def __init__(self, name, sim_time):
        assert type(name) == str, ('name must be a string. input value is a', type(name))
        assert type(sim_time) == int, ('sim_time must be an integer. input value is a', type(sim_time))
        assert sim_time > 0, 'sim_time must be greater than zero'
        self.name = name
        self.month = int(0)

        print(' ================================ \n', self.name, 'created \n ================================')
        return


class PET_Manufacturer(Agent):
    # object initialisation
    def __init__(self, name, sim_time):
        super().__init__(name, sim_time)

        # define independent variables for current time
        self.production_volume = Parameter(production_volume, init=1000)
        # total PET production per annum, starts at 1000
        self.unit_sale_price = Parameter(unit_sale_price)  # sale price of one unit of PET
        self.unit_feedstock_cost = Parameter(unit_feedstock_cost)  # feedstock cost per unit of PET produced
        self.unit_process_cost = Parameter(unit_process_cost)  # cost of running process per unit of PET produced

        self.bio_feedstock_cost = Parameter(bio_feedstock_cost)  # bio feedstock cost per unit of PET produced
        self.bio_process_cost = Parameter(
            bio_process_cost)  # cost of process per unit of PET from bio routes, starts at 1.5
        self.proportion_bio = Parameter(proportion_bio)  # proportion of production from biological feedstocks

        # list of all independent variables, listed in the order in which they must be computed (if it matters at all)
        self.independent_variables = [
            self.production_volume,
            self.unit_sale_price,
            self.unit_feedstock_cost,
            self.unit_process_cost,
            self.bio_feedstock_cost,
            self.bio_process_cost,
            self.proportion_bio
        ]

        # define dependent variables
        self.gross_profit = Parameter(gross_profit)  # profits prior to taxes and levies
        self.emissions = Parameter(emissions)  # emissions from manufacturing PET from fossil fuels
        self.tax_payable = Parameter(tax_payable)
        self.levies_payable = Parameter(levies_payable)
        self.net_profit = Parameter(net_profit)  # monthly profit after tax and levies
        self.profitability = Parameter(profitability)  # profitability (net profit per unit production)

        # list of all parametrised dependent variables, listed in the order in which they must be computed
        self.dependent_variables = [
            self.emissions,
            self.levies_payable,
            self.gross_profit,
            self.tax_payable,
            self.net_profit,
            self.profitability
        ]

        # now define other parameters which will not be recorded or projected
        self.projection_time = 120  # how many months into the future will be predicted?

        self.proportion_bio_target = np.float64()  # target value for the proportion of production via bio route
        self.projection_met = int(0)  # 1 or 0 depending on whether the next target will be met by current
        # projection

        self.tax_rate = np.float64(0.19)  # current tax on profits, starts at 19%
        self.levy_rate = np.float64(0.2)  # current levy on production/emission/consumption/etc.
        self.emissions_rate = np.float64(5)  # units of emissions per unit of PET produced from non-bio route

        self.levy_rate_changing = False
        self.time_to_levy_change = int()
        self.future_levy_rate = np.float64()

        # additional projection variables
        self.tax_rate_projection = np.zeros(self.projection_time)
        self.levy_projection = np.zeros(self.projection_time)

        # define variables for the targets against which projections are measured
        # and the times at which they happen
        self.target1_value = np.float64(1.5)  # currently fixed values
        self.target1_year = 5

        self.target2_value = np.float64(1.5)  # currently fixed values
        self.target2_year = 10

        self.beyond_target_range = False  # a boolean set to true if the simulation runs beyond the point for which
        # targets are defined

        self.proportion_change_rate = np.float64(0.1 / 15)  # greatest possible monthly change in self.proportion_bio
        self.implementation_delay = int(3)  # time delay between investment decision and movement of bio_proportion
        self.implementation_countdown = int(0)  # countdown to change of direction
        self.under_construction = False  # is change in bio capacity occurring?

        # output initialisation state to console
        print(' INITIAL STATE \n -------------'
              '\n Annual production volume:', self.production_volume.value,
              '\n Projection horizon (months):', self.projection_time,
              '\n Target at year', self.target1_year, ':', self.target1_value,
              '\n Target at year', self.target2_year, ':', self.target2_value,
              '\n -------------')

        # run_check()

        return

    def update_independent_variables(self):
        # calculate new values for all independent variables
        for parameter in self.independent_variables:
            parameter.update(self)

        return

    def update_dependent_variables(self):
        for parameter in self.dependent_variables:
            parameter.update(self)
        return

    def record_timestep(self):
        # method to write current variables (independent and dependent) to records
        for variable in self.independent_variables:
            variable.record(self.month)

        for variable in self.dependent_variables:
            variable.record(self.month)
        return

    def update_current_state(self):
        # methods to be called every time the month is advanced
        self.update_independent_variables()
        self.update_dependent_variables()
        return

    # region -- methods for making projections into the future
    def project_volume(self):
        # calculates the projected (annualised) PET production volume for each month,
        # recording it to self.production_projection
        predicted_annual_growth_rate = 1.02
        monthly_growth_rate = np.power(predicted_annual_growth_rate, 1 / 12)
        initial_volume = self.production_volume.value
        # calculated using a fixed month-on-month growth rate from the most recent production volume
        for i in range(self.projection_time):
            self.production_volume.projection[i] = initial_volume * pow(monthly_growth_rate, i)

        return

    def project_sale_price(self):
        # Calculate the projected PET sale prices
        self.unit_sale_price.projection.fill(6)  # fixed value (mean of normal dist from self.refresh_unit_sale_price)
        return

    def project_feedstock_cost(self):
        # Calculate the projected PET feedstock costs
        self.unit_feedstock_cost.projection.fill(2)  # fixed value (mean of normal dist from self.refresh_...)
        return

    def project_process_cost(self):
        # Calculate the projected PET processing costs
        self.unit_process_cost.projection.fill(1)  # fixed value (mean of normal dist from self.refresh_...)
        return

    def project_proportion_bio(self):
        # projection of proportion of production from bio routes
        time_to_target = int(np.ceil((self.proportion_bio_target - self.proportion_bio.value) /
                                     self.proportion_change_rate)
                             + self.implementation_countdown)
        self.proportion_bio.projection.fill(self.proportion_bio_target)

        if time_to_target > 1:

            for i in range(time_to_target - 1):
                try:
                    self.proportion_bio.projection[i] = self.proportion_bio.value + self.proportion_change_rate * \
                                                        (i + 1)
                except IndexError:
                    print('time to reach target bio proportion is longer than', self.projection_time, 'months')
                    print('behaviour in these conditions is undefined. aborting simulation')
                    raise SystemExit(0)
        else:
            pass
        return

    def project_bio_feedstock_cost(self):
        self.bio_feedstock_cost.projection.fill(2)
        return

    def project_bio_process_cost(self):
        self.bio_process_cost.projection.fill(1.05)
        return

    def project_emissions(self):
        monthly_production_projection = self.production_volume.projection / 12
        self.emissions.projection = np.multiply(
            monthly_production_projection, np.subtract(
                np.ones(self.projection_time), self.proportion_bio.projection)
        ) * self.emissions_rate
        return

    def project_gross_profit(self):
        # calculate revenues and costs at each month
        monthly_production_projection = self.production_volume.projection / 12

        revenue_projection = np.multiply(monthly_production_projection, self.unit_sale_price.projection)

        fossil_cost_projection = np.multiply(
            np.add(np.multiply(monthly_production_projection, self.unit_feedstock_cost.projection),
                   np.multiply(monthly_production_projection, self.unit_process_cost.projection)),
            np.subtract(np.ones(self.projection_time), self.proportion_bio.projection))
        bio_cost_projection = np.multiply(
            np.add(np.multiply(monthly_production_projection, self.bio_feedstock_cost.projection),
                   np.multiply(monthly_production_projection, self.bio_process_cost.projection)),
            self.proportion_bio.projection)

        total_cost_projection = np.add(fossil_cost_projection, bio_cost_projection)

        self.gross_profit.projection = np.subtract(revenue_projection, total_cost_projection)
        return

    def project_tax_rate(self):
        self.tax_rate_projection = np.ones(self.projection_time) * self.tax_rate
        return

    def project_levy_rate(self):
        if not self.levy_rate_changing:
            self.levy_projection.fill(self.levy_rate)
        else:
            self.levy_projection.fill(self.future_levy_rate)
            for i in range(self.time_to_levy_change):
                self.levy_projection[i] = self.levy_rate
        return

    def project_tax_payable(self):
        self.tax_payable.projection = np.multiply(self.gross_profit.projection, self.tax_rate_projection)
        return

    def project_levies_payable(self):
        self.levies_payable.projection = np.multiply(self.emissions.projection, self.levy_projection)
        return

    def project_net_profit(self):
        p_0 = self.gross_profit.projection
        p_1 = np.subtract(p_0, self.tax_payable.projection)
        p_2 = np.subtract(p_1, self.levies_payable.projection)
        self.net_profit.projection = p_2
        return

    def project_profitability(self):
        self.profitability.projection = np.divide(self.net_profit.projection, self.production_volume.projection / 12)

    def project_independents(self):
        # calculate projections for independent variables
        self.project_volume()
        self.project_sale_price()
        self.project_feedstock_cost()
        self.project_process_cost()
        self.project_proportion_bio()
        self.project_bio_feedstock_cost()
        self.project_bio_process_cost()
        self.project_levy_rate()
        self.project_tax_rate()
        return

    def project_dependents(self):
        # calculate projections for dependent variables (i.e. must run after self.project_independents)
        # order of operations for these methods may be important - CHECK
        self.project_gross_profit()
        self.project_emissions()
        self.project_tax_payable()
        self.project_levies_payable()
        self.project_net_profit()
        self.project_profitability()
        return

    def new_projection(self):
        self.project_independents()
        self.project_dependents()
        return

    # endregion

    def projection_check(self):
        # checks whether the next target will be met on the basis of latest projection
        # monthly profit targets at years 5 and 10

        time_to_yr1 = self.target1_year * 12 - self.month
        time_to_yr2 = self.target2_year * 12 - self.month

        if time_to_yr1 > 0:
            if self.profitability.projection[time_to_yr1] >= self.target1_value:
                self.projection_met = 1
            else:
                self.projection_met = 0

        elif time_to_yr2 > 0:
            if self.profitability.projection[time_to_yr2] >= self.target2_value:
                self.projection_met = 1
            else:
                self.projection_met = 0

        else:
            # if there is no target defined into the future, PROJECTION_MET is set to 0
            self.projection_met = 0

            if not self.beyond_target_range:
                print('No target defined beyond month', self.month)
                self.beyond_target_range = True

        return

    def investment_decision(self):
        # decision logic for increasing investment in biological process route
        if self.projection_met == 1:
            pass
        elif self.projection_met == 0:
            while self.proportion_bio_target <= 0.9 and self.projection_met == 0:
                self.implementation_countdown = self.implementation_delay
                self.proportion_bio_target += 0.1
                self.new_projection()
                self.projection_check()
                if self.proportion_bio_target == 1 and self.projection_met == 0:
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

        self.update_current_state()
        if self.month % 12 == 0:
            self.new_projection()
            self.projection_check()
            self.investment_decision()
        self.record_timestep()
        return
