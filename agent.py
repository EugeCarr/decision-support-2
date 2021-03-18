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
    # an object class with a current value, a record of all past values, and a variable to hold a projection
    # only compatible with data types which can be parsed as a float
    # argument FUN is a function taking a single AGENT object as an argument and returns the next value of the variable
    def __init__(self, fun, project, history_time=120, projection_time=120, init=np.float64(0)):
        assert type(history_time) == int
        assert history_time > 0
        assert type(projection_time) == int
        assert projection_time > 0

        self.fun = fun
        self.project = project

        self.value = np.float64(init)

        self.projection = np.zeros(projection_time)
        self.history = np.zeros(history_time)

        return

    def update(self, agent):
        # calls the defined update function to calculate the next value of the variable
        assert isinstance(agent, Agent)
        self.value = self.fun(agent)
        return

    def record(self, time):
        # writes the current value of the parameter to a chosen element of the record array
        self.history[time] = self.value
        return

    def forecast(self, agent):
        assert isinstance(agent, Agent)
        self.projection = self.project(agent)
        return


# region -- parameter calculation functions

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
    growth_rate_1 = 1.02  # YoY growth rate for the second simulation period, expressed as a ratio
    growth_rate_1_monthly = np.power(growth_rate_1, 1 / 12)  # annual growth rate changed to month-on-month

    if month == 0:
        val = volume

    elif month <= sim_period_0_months:
        val = volume * growth_rate_0_monthly

    else:
        val = volume * growth_rate_1_monthly

    return val


def unit_sale_price(agent) -> np.float64:
    # unit sale price is given by a normal distribution
    mean = np.float64(5)
    std_dev = 0.01
    val = np.float64(np.random.normal(mean, std_dev, None))
    return val


def unit_feedstock_cost(agent) -> np.float64:
    # unit feedstock cost is given by a normal distribution
    mean = np.float64(2)
    std_dev = 0.01
    val = np.float64(np.random.normal(mean, std_dev, None))
    return val


def unit_process_cost(agent) -> np.float64:
    # process cost is given by a normal distribution around a mean which is a weakly decreasing function of
    # production volume, such that a doubling in production reduces processing unit cost by 10%, starting from 1
    mean = 2.8576 / np.power(agent.production_volume.value, 0.152)
    std_dev = 0.005
    val = np.float64(np.random.normal(mean, std_dev, None))
    return val


def bio_feedstock_cost(agent) -> np.float64:
    # unit feedstock cost is given by a normal distribution
    mean = np.float64(2)
    std_dev = 0.01
    val = np.float64(np.random.normal(mean, std_dev, None))
    return val


def bio_process_cost(agent) -> np.float64:
    # process cost is given by a normal distribution
    mean = 1.05
    std_dev = 0.005
    val = np.float64(np.random.normal(mean, std_dev, None))
    return val


def proportion_bio(agent) -> np.float64:
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


def levy_rate(agent) -> np.float64:
    val = agent.levy
    return val


def emissions(agent) -> np.float64:
    fossil_production = agent.production_volume.value / 12 * (1 - agent.proportion_bio.value)
    val = fossil_production * agent.emissions_rate
    return val


def levies_payable(agent) -> np.float64:
    """This will calculate the levies payable on production/consumption/emission, once they are defined"""
    val = agent.levy * agent.emissions.value
    return val


def gross_profit(agent) -> np.float64:
    production_in_month = agent.production_volume.value / 12
    revenue = production_in_month * agent.unit_sale_price.value
    costs = (production_in_month * ((1 - agent.proportion_bio.value) *
                                    (agent.unit_feedstock_cost.value + agent.unit_process_cost.value) +
                                    agent.proportion_bio.value *
                                    (agent.bio_feedstock_cost.value + agent.bio_process_cost.value))
             + agent.levies_payable.value)
    val = revenue - costs
    return val


def tax_payable(agent) -> np.float64:
    val = agent.gross_profit.value * agent.tax_rate
    return val


def net_profit(agent) -> np.float64:
    val = agent.gross_profit.value - agent.tax_payable.value
    return val


def profitability(agent) -> np.float64:
    val = agent.net_profit.value / (agent.production_volume.value / 12)
    return val


def expansion_cost(agent) -> np.float64:
    val = np.float64()
    if agent.month > 0:
        bio_increase = agent.bio_capacity.value - agent.bio_capacity.history[agent.month - 1]
        bio_cost = agent.bio_capacity_cost * bio_increase

        fossil_increase = agent.fossil_capacity.value - agent.fossil_capacity.history[agent.month - 1]
        fossil_cost = agent.fossil_capacity_cost * fossil_increase
        val = np.float64(bio_cost + fossil_cost)
    elif agent.month == 0:
        val = 0

    return val


def bio_capacity(agent) -> np.float64:
    # only ever increases.
    val = np.float64()
    if agent.month > 0:
        prev = agent.bio_capacity.history[agent.month - 1]
        now = agent.production_volume.value * agent.proportion_bio.value
        val = max(now, prev)
    elif agent.month == 0:
        val = 0
    else:
        pass
    return val


def fossil_capacity(agent) -> np.float64:
    val = np.float64()
    if agent.month > 0:
        prev = agent.fossil_capacity.history[agent.month - 1]
        now = agent.production_volume.value * (1 - agent.proportion_bio.value)
        val = max(now, prev)
    elif agent.month == 0:
        val = 1000
    else:
        pass
    return val


def liquidity(agent) -> np.float64:
    val = agent.liquidity.value + agent.net_profit.value - agent.expansion_cost.value
    return val


# endregion

# region -- projection functions

def production_volume_projection(agent) -> np.ndarray:
    # calculates the projected (annualised) PET production volume for each month,
    # recording it to self.production_projection
    predicted_annual_growth_rate = 1.02
    monthly_growth_rate = np.power(predicted_annual_growth_rate, 1 / 12)
    initial_volume = agent.production_volume.value
    # calculated using a fixed month-on-month growth rate from the most recent production volume

    proj = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        proj[i] = initial_volume * pow(monthly_growth_rate, i)

    return proj


def unit_sale_price_projection(agent) -> np.ndarray:
    # Calculate the projected PET sale prices
    proj = np.zeros(agent.projection_time)
    proj.fill(5)  # fixed value
    return proj


def unit_feedstock_cost_projection(agent) -> np.ndarray:
    # Calculate the projected PET feedstock costs
    proj = np.zeros(agent.projection_time)
    proj.fill(2)  # fixed value
    return proj


def unit_process_cost_projection(agent) -> np.ndarray:
    # Calculate the projected PET processing costs
    proj = np.zeros(agent.projection_time)
    proj.fill(1)  # fixed value
    return proj


def proportion_bio_projection(agent) -> np.ndarray:
    # projection of proportion of production from bio routes
    proj = np.zeros(agent.projection_time)
    time_to_target = int(np.ceil((agent.proportion_bio_target - agent.proportion_bio.value) /
                                 agent.proportion_change_rate)
                         + agent.implementation_countdown)
    proj.fill(agent.proportion_bio_target)

    if time_to_target > 1:

        for i in range(time_to_target - 1):
            try:
                proj[i] = agent.proportion_bio.value + agent.proportion_change_rate * \
                          (i + 1)
            except IndexError:
                print('time to reach target bio proportion is longer than', agent.projection_time, 'months')
                print('behaviour in these conditions is undefined. aborting simulation')
                raise SystemExit(0)
    else:
        pass
    return proj


def bio_feedstock_cost_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    proj.fill(2)
    return proj


def bio_process_cost_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    proj.fill(1.05)
    return proj


def emissions_projection(agent) -> np.ndarray:
    monthly_production_projection = agent.production_volume.projection / 12
    proj = np.multiply(
        monthly_production_projection, np.subtract(
            np.ones(agent.projection_time), agent.proportion_bio.projection)
    ) * agent.emissions_rate
    return proj


def levies_payable_projection(agent) -> np.ndarray:
    proj = np.multiply(agent.emissions.projection, agent.levy_rate.projection)
    return proj


def gross_profit_projection(agent) -> np.ndarray:
    # calculate revenues and costs at each month
    monthly_production_projection = agent.production_volume.projection / 12

    revenue_projection = np.multiply(monthly_production_projection, agent.unit_sale_price.projection)

    fossil_cost_projection = np.multiply(
        np.add(np.multiply(monthly_production_projection, agent.unit_feedstock_cost.projection),
               np.multiply(monthly_production_projection, agent.unit_process_cost.projection)),
        np.subtract(np.ones(agent.projection_time), agent.proportion_bio.projection))
    bio_cost_projection = np.multiply(
        np.add(np.multiply(monthly_production_projection, agent.bio_feedstock_cost.projection),
               np.multiply(monthly_production_projection, agent.bio_process_cost.projection)),
        agent.proportion_bio.projection)

    total_cost_projection = np.add(
        np.add(fossil_cost_projection, bio_cost_projection),
        agent.levies_payable.projection)

    proj = np.subtract(revenue_projection, total_cost_projection)
    return proj


def tax_rate_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.tax_rate
    return proj


def levy_rate_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    if not agent.levy_rate_changing:
        proj.fill(agent.levy)
    else:
        proj.fill(agent.future_levy_rate)
        for i in range(agent.time_to_levy_change):
            proj[i] = agent.levy
    return proj


def tax_payable_projection(agent) -> np.ndarray:
    proj = np.multiply(agent.gross_profit.projection, agent.tax_rate_projection)
    return proj


def net_profit_projection(agent) -> np.ndarray:
    profit_g = agent.gross_profit.projection
    proj = np.subtract(profit_g, agent.tax_payable.projection)
    return proj


def profitability_projection(agent) -> np.ndarray:
    proj = np.divide(agent.net_profit.projection, agent.production_volume.projection / 12)
    return proj


def expansion_cost_projection(agent) -> np.ndarray:
    bio_expansion = np.zeros(agent.projection_time)
    fossil_expansion = np.zeros(agent.projection_time)
    if agent.month > 0:
        bio_expansion[0] = agent.bio_capacity.value - agent.bio_capacity.history[agent.month - 1]
        fossil_expansion[0] = agent.fossil_capacity.value - agent.fossil_capacity.history[agent.month - 1]
    elif agent.month == 0:
        bio_expansion[0] = 0
        fossil_expansion[0] = 0
    else:
        pass

    for i in range(1, agent.projection_time):
        bio_expansion[i] = agent.bio_capacity.projection[i] - agent.bio_capacity.projection[i - 1]
        fossil_expansion[i] = agent.fossil_capacity.projection[i] - agent.fossil_capacity.projection[i - 1]

    bio_expansion_cost = bio_expansion * agent.bio_capacity_cost
    fossil_expansion_cost = fossil_expansion * agent.fossil_capacity_cost

    proj = np.add(bio_expansion_cost, fossil_expansion_cost)

    return proj


def bio_capacity_projection(agent) -> np.ndarray:
    bio_production = np.multiply(agent.production_volume.projection, agent.proportion_bio.projection)
    proj = np.zeros(agent.projection_time)
    proj[0] = max(agent.bio_capacity.value, bio_production[0])
    for i in range(1, agent.projection_time):
        proj[i] = max(bio_production[i], proj[i - 1])
    return proj


def fossil_capacity_projection(agent) -> np.ndarray:
    fossil_production = np.multiply(agent.production_volume.projection,
                                    np.subtract(np.ones(agent.projection_time), agent.proportion_bio.projection))
    proj = np.zeros(agent.projection_time)
    proj[0] = max(agent.fossil_capacity.value, fossil_production[0])
    for i in range(1, agent.projection_time):
        proj[i] = max(fossil_production[i], proj[i - 1])
    return proj


def liquidity_projection(agent) -> np.ndarray:
    liq = np.zeros(agent.projection_time)
    liq.fill(agent.liquidity.value)

    revenues = np.cumsum(agent.net_profit.projection, dtype=np.float64)
    costs = np.cumsum(agent.expansion_cost.projection, dtype=np.float64)
    profits = np.subtract(revenues, costs)
    proj = np.add(liq, profits)
    return proj


# endregion

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
        """To add a new parameter, define it as an attribute of object type Parameter. Then add it to the correct list
        of variables (dependent or independent) in the correct place so parameters are computed in the right order. 
        The argument fun is the function which, given an argument of object type Agent, returns a single float which
        is the next value of the parameter. 
        The argument project is a function which, given an argument of object type Agent, returns an array of size n
        which is the projected value of the parameter for the next n months.
        Optional argument init is the initial value of the parameter. This is usually unnecessary.
        history_time and projection_time should only be changed from their default values with great care."""
        # define independent variables for current time
        self.production_volume = Parameter(production_volume, production_volume_projection, init=np.float64(1000))
        # total PET production per annum, starts at 1000
        self.unit_sale_price = Parameter(unit_sale_price, unit_sale_price_projection)  # sale price of one unit of PET
        self.unit_feedstock_cost = Parameter(unit_feedstock_cost, unit_feedstock_cost_projection)
        # feedstock cost per unit of PET produced
        self.unit_process_cost = Parameter(unit_process_cost, unit_process_cost_projection)
        # cost of running process per unit of PET produced

        self.bio_feedstock_cost = Parameter(bio_feedstock_cost, bio_feedstock_cost_projection)
        # bio feedstock cost per unit of PET produced
        self.bio_process_cost = Parameter(bio_process_cost, bio_process_cost_projection)
        # cost of process per unit of PET from bio routes
        self.proportion_bio = Parameter(proportion_bio, proportion_bio_projection)
        # proportion of production from biological feedstocks
        self.levy_rate = Parameter(levy_rate, levy_rate_projection, init=np.float64(0.2))

        self.bio_capacity = Parameter(bio_capacity, bio_capacity_projection)
        self.fossil_capacity = Parameter(fossil_capacity, fossil_capacity_projection, init=np.float64(1000))

        # list of all independent variables, listed in the order in which they must be computed (if it matters at all)
        self.independent_variables = [
            self.production_volume,
            self.unit_sale_price,
            self.unit_feedstock_cost,
            self.unit_process_cost,
            self.bio_feedstock_cost,
            self.bio_process_cost,
            self.proportion_bio,
            self.levy_rate,
            self.bio_capacity,
            self.fossil_capacity
        ]

        # define dependent variables
        self.expansion_cost = Parameter(expansion_cost, expansion_cost_projection)
        # cost of increasing production capacity
        self.gross_profit = Parameter(gross_profit, gross_profit_projection)  # profits after levies and before taxes
        self.emissions = Parameter(emissions, emissions_projection)
        # emissions from manufacturing PET from fossil fuels
        self.tax_payable = Parameter(tax_payable, tax_payable_projection)
        self.levies_payable = Parameter(levies_payable, levies_payable_projection)
        self.net_profit = Parameter(net_profit, net_profit_projection)  # monthly profit after tax and levies
        self.profitability = Parameter(profitability, profitability_projection)
        # profitability (net profit per unit production)
        self.liquidity = Parameter(liquidity, liquidity_projection, init=np.float64(1000))  # accumulated cash

        # list of all parametrised dependent variables, listed in the order in which they must be computed
        self.dependent_variables = [
            self.expansion_cost,
            self.emissions,
            self.levies_payable,
            self.gross_profit,
            self.tax_payable,
            self.net_profit,
            self.profitability,
            self.liquidity
        ]

        # now define other parameters which will not be recorded or projected
        self.projection_time = 120  # how many months into the future will be predicted?

        self.proportion_bio_target = np.float64()  # target value for the proportion of production via bio route
        self.projection_met = False  # 1 or 0 depending on whether the next target will be met by current
        # projection

        self.tax_rate = np.float64(0.19)  # current tax on profits, starts at 19%
        self.levy = np.float(0.2)
        self.emissions_rate = np.float64(5)  # units of emissions per unit of PET produced from non-bio route

        self.levy_rate_changing = False
        self.time_to_levy_change = int()
        self.future_levy_rate = np.float64()

        # additional projection variables
        self.tax_rate_projection = np.ones(self.projection_time) * self.tax_rate
        self.levy_projection = np.zeros(self.projection_time)

        # define variables for the targets against which projections are measured
        # and the times at which they happen
        self.target1_value = np.float64(0.8)  # currently fixed values
        self.target1_year = 5

        self.target2_value = np.float64(1.0)  # currently fixed values
        self.target2_year = 10

        self.beyond_target_range = False  # a boolean set to true if the simulation runs beyond the point for which
        # targets are defined

        self.proportion_change_rate = np.float64(0.1 / 9)  # greatest possible monthly change in self.proportion_bio
        self.implementation_delay = int(3)  # time delay between investment decision and movement of bio_proportion
        self.implementation_countdown = int(0)  # countdown to change of direction
        self.under_construction = False  # is change in bio capacity occurring?

        self.fossil_capacity_cost = np.float64(10)  # one-time cost of increasing production capacity for fossil route
        self.bio_capacity_cost = np.float64(12)  # one-time cost of increasing production capacity for bio route

        # output initialisation state to console
        print(' INITIAL STATE \n -------------'
              '\n Annual production volume:', self.production_volume.value,
              '\n Projection horizon (months):', self.projection_time,
              '\n Target at year', self.target1_year, ':', self.target1_value,
              '\n Target at year', self.target2_year, ':', self.target2_value,
              '\n -------------')

        # run_check()

        return

    # region -- methods for updating values and projections
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

    def project_independents(self):
        # calculate projections for independent variables
        for parameter in self.independent_variables:
            parameter.forecast(self)
        return

    def project_dependents(self):
        # calculate projections for dependent variables (i.e. must run after self.project_independents)
        # order of operations for these methods may be important - CHECK
        for parameter in self.dependent_variables:
            parameter.forecast(self)

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
                self.projection_met = True
            else:
                self.projection_met = False

        elif time_to_yr2 > 0:
            if self.profitability.projection[time_to_yr2] >= self.target2_value:
                self.projection_met = True
            else:
                self.projection_met = False

        else:
            # if there is no target defined into the future, PROJECTION_MET is set to false
            self.projection_met = False

            if not self.beyond_target_range:
                print('No target defined beyond month', self.month)
                self.beyond_target_range = True

        return

    def investment_decision(self):
        # decision logic for increasing investment in biological process route
        if self.projection_met:
            pass
        elif not self.projection_met:
            while self.proportion_bio_target <= 0.9 and not self.projection_met:
                self.implementation_countdown = self.implementation_delay
                self.proportion_bio_target += 0.1
                self.new_projection()
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

        self.update_current_state()
        if self.month % 12 == 0:
            self.new_projection()
            self.projection_check()
            self.investment_decision()
        self.record_timestep()
        return
