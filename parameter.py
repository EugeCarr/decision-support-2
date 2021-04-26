import numpy as np
import agent as ag
import copy
from agent import run_check
from scipy import optimize
from scipy.optimize import Bounds


class Environment_Variable(object):
    # shared variables, such as market prices and aggregated demand, are stored as environment variables
    def __init__(self, fun, sim_length, init=np.float64(0)):
        assert type(sim_length) == int
        assert sim_length > 0

        self.fun = fun

        self.value = np.float64(init)

        self.history = np.zeros(sim_length)

        return

    def update(self, environment):
        # calls the defined update function to calculate the next value of the variable
        assert isinstance(environment, ag.Environment)
        self.value = np.float64(self.fun(environment))
        return

    def record(self, month):
        # writes the current value of the parameter to a chosen element of the record array
        self.history[month] = self.value
        return


def pet_price(env) -> np.float64:
    # pet price is a random walk from the initial value
    val = env.parameter['pet_price'].value
    std_dev = 0.01
    deviation = np.float64(np.random.normal(0, std_dev, None))
    val += deviation

    # val = env.parameter['pet_price'].history[0]
    return val


def fossil_feedstock_price(env) -> np.float64:
    # price is a random walk from the initial value
    val = env.parameter['fossil_feedstock_price'].value
    std_dev = 0.01
    deviation = np.float64(np.random.normal(0, std_dev, None))
    val += deviation

    # val = env.parameter['pet_price'].history[0]
    return val


def bio_feedstock_price(env) -> np.float64:
    # price is a random walk from the initial value
    val = env.parameter['bio_feedstock_price'].value
    # std_dev = 0.01
    # deviation = np.float64(np.random.normal(0, std_dev, None))
    # val += deviation

    # val = env.parameter['pet_price'].history[0]
    return val


def levy_rate(env) -> np.float64:
    return env.parameter['levy_rate'].value


def demand(env) -> np.float64:
    current = env.parameter['demand'].value

    # demand is defined by constant growth rate, plus some randomness
    growth_rate = 1.02  # YoY growth rate, expressed as a ratio
    growth_rate_monthly = np.power(growth_rate, 1 / 12)  # annual growth rate changed to month-on-month
    val = current * growth_rate_monthly
    std_dev = 1
    random = np.float64(np.random.normal(0, std_dev, None))
    val += random

    if env.month == 0:
        val = current

    return val


class Parameter(object):
    # an object class with a current value, a record of all past values, and a variable to hold a projection
    # only compatible with data types which can be parsed as a float

    """The argument fun is the function which, given an argument of object type Agent, returns a single float which
        is the next value of the parameter.
        The argument project is a function which, given an argument of object type Agent, returns an array
        which is the projected value of the parameter for the next n months.
        Optional argument init is the initial value of the parameter. Default value zero.
        sim_length is the number of months the simulation will run for, used to initialise records of the correct length
        projection_time should only be changed from the default with great care."""

    def __init__(self, fun, project, sim_length, projection_time=120, init=np.float64(0)):
        assert type(sim_length) == int
        assert sim_length > 0
        assert type(projection_time) == int
        assert projection_time > 0

        self.projection_time = projection_time

        self.fun = fun
        self.project = project

        self.value = np.float64(init)

        self.projection = np.zeros(projection_time)
        self.history = np.zeros(sim_length)

        return

    def update(self, agent):
        # calls the defined update function to calculate the next value of the variable
        assert isinstance(agent, ag.Agent)
        self.value = np.float64(self.fun(agent))
        return

    def record(self, time):
        # writes the current value of the parameter to a chosen element of the record array
        self.history[time] = self.value
        return

    def forecast(self, agent):
        assert isinstance(agent, ag.Agent)
        array = self.project(agent)
        self.projection = array
        return


def blank(agent):
    # an empty function intended for use when a parameter needs to be projected by an agent
    # but is calculated in the environment. exists to satisfy argument requirements of Parameter object
    pass


# def production_volume(agent) -> np.float64:
#     volume = agent.parameter['production_volume'].value
#
#     # production volume is defined by growth rates
#
#     growth_rate = 1.02  # YoY growth rate for the simulation period, expressed as a ratio
#     growth_rate_monthly = np.power(growth_rate, 1 / 12)  # annual growth rate changed to month-on-month
#     target_amount = volume * growth_rate_monthly
#
#     if agent.month == 0:
#         val = volume
#
#     else:
#         val = target_amount
#
#     return val


def demand_projection(agent) -> np.ndarray:
    current = agent.env.parameter['demand'].value
    growth_rate = 1.02  # YoY growth rate for the simulation period, expressed as a ratio
    growth_rate_monthly = np.power(growth_rate, 1 / 12)  # annual growth rate changed to month-on-month

    proj = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        proj[i] = current * pow(growth_rate_monthly, i + 1)

    return proj


def fossil_process_cost(agent) -> np.float64:
    # normal distribution
    if agent.month == 0:
        mean = agent.parameter['fossil_process_cost'].value
    else:
        mean = agent.parameter['fossil_process_cost'].history[0]
    std_dev = 0
    deviation = np.float64(np.random.normal(0, std_dev, None))
    val = mean + deviation

    return val


def bio_process_cost(agent) -> np.float64:
    # normal distribution
    if agent.month == 0:
        mean = agent.parameter['bio_process_cost'].value
    else:
        mean = agent.parameter['bio_process_cost'].history[0]
    std_dev = 0
    deviation = np.float64(np.random.normal(0, std_dev, None))
    val = mean + deviation

    return val


def proportion_bio(agent) -> np.float64:
    # monthly change in bio proportion is either the amount to reach the target value, or else the maximum change
    val = np.float64(agent.parameter['proportion_bio'].value)
    if agent.implementation_countdown == 0 and agent.parameter['proportion_bio'].value != agent.proportion_bio_target:
        agent.under_construction = True
        distance_from_target = agent.proportion_bio_target - agent.parameter['proportion_bio'].value
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


def emissions(agent) -> np.float64:
    val = agent.parameter['fossil_production'].value / 12 * agent.emissions_rate
    return val


def levies_payable(agent) -> np.float64:
    """This will calculate the levies payable on production/consumption/emission, once they are defined"""
    val = agent.env.parameter['levy_rate'].value * agent.parameter['emissions'].value
    return val


def gross_profit(agent) -> np.float64:
    # revenue calculation
    production_in_month = (agent.parameter['fossil_production'].value + agent.parameter['bio_production'].value) / 12
    sellable_amount = agent.env.parameter['demand'].value
    product_sold = min(sellable_amount, production_in_month)
    revenue = product_sold * agent.env.parameter['pet_price'].value

    # costs calculation
    costs = (
            agent.parameter['fossil_feedstock_consumption'].value *
            agent.env.parameter['fossil_feedstock_price'].value +

            agent.parameter['bio_feedstock_consumption'].value *
            agent.env.parameter['bio_feedstock_price'].value +

            production_in_month * (
                    (1 - agent.parameter['proportion_bio'].value) *
                    agent.parameter['fossil_process_cost'].value +

                    agent.parameter['proportion_bio'].value *
                    agent.parameter['bio_process_cost'].value
            ) +

            agent.parameter['levies_payable'].value +

            agent.capacity_maintenance_cost * (
                    agent.parameter['fossil_capacity'].value + agent.parameter['bio_capacity'].value
            )
    )

    val = revenue - costs
    return val


def tax_payable(agent) -> np.float64:
    val = agent.parameter['gross_profit'].value * agent.tax_rate
    return val


def net_profit(agent) -> np.float64:
    val = agent.parameter['gross_profit'].value - agent.parameter['tax_payable'].value
    return val


def profitability(agent) -> np.float64:
    val = agent.parameter['net_profit'].value / (agent.parameter['total_production'].value / 12)
    return val


def expansion_cost(agent) -> np.float64:
    val = np.float64()
    if agent.month > 0:
        bio_increase = (agent.parameter['bio_capacity'].value -
                        agent.parameter['bio_capacity'].history[agent.month - 1])
        bio_cost = agent.bio_capacity_cost * bio_increase

        fossil_increase = (agent.parameter['fossil_capacity'].value -
                           agent.parameter['fossil_capacity'].history[agent.month - 1])
        fossil_cost = agent.fossil_capacity_cost * fossil_increase
        val = np.float64(bio_cost + fossil_cost)
    elif agent.month == 0:
        val = 0

    return val


# def bio_capacity(agent) -> np.float64:
#     # only ever increases.
#     val = np.float64()
#     if agent.month > 0:
#         prev = agent.parameter['bio_capacity'].history[agent.month - 1]
#         now = agent.parameter['production_volume'].value * agent.parameter['proportion_bio'].value
#         val = max(now, prev)
#     elif agent.month == 0:
#         val = 0
#     else:
#         pass
#     return val
#
#
# def fossil_capacity(agent) -> np.float64:
#     val = np.float64()
#     if agent.month > 0:
#         prev = agent.parameter['fossil_capacity'].history[agent.month - 1]
#         now = agent.parameter['production_volume'].value * (1 - agent.parameter['proportion_bio'].value)
#         val = max(now, prev)
#     elif agent.month == 0:
#         val = 1000
#     else:
#         pass
#     return val


def liquidity(agent) -> np.float64:
    val = (agent.parameter['liquidity'].value +
           agent.parameter['net_profit'].value -
           agent.parameter['expansion_cost'].value)

    if val < 0 and not agent.negative_liquidity:
        agent.negative_liquidity = True
        print('Liquidity went negative in month', agent.month)

    if agent.negative_liquidity and val > 0:
        agent.negative_liquidity = False
        print('Liquidity went positive in month', agent.month)

    return val


def profit_margin(agent) -> np.float64:
    production_in_month = agent.parameter['total_production'].value / 12
    val = agent.parameter['net_profit'].value / (production_in_month * agent.env.parameter['pet_price'].value)
    return val


def bio_capacity_max(agent) -> np.float64:
    baseline_capacity = agent.parameter['total_production'].history[0]
    max_cap = baseline_capacity * np.power(1.02, agent.sim_time) * 2
    # defines an arbitrary maximum capacity as the starting production in month 0 multiplied by two after growth
    return max_cap


def fossil_capacity_max(agent) -> np.float64:
    baseline_capacity = agent.parameter['total_production'].history[0]
    max_cap = baseline_capacity * np.power(1.02, agent.sim_time) * 2
    return max_cap


def production_volume_projection(agent) -> np.ndarray:
    # calculates the projected (annualised) PET production volume for each month,
    # recording it to self.production_projection
    predicted_annual_growth_rate = 1.02
    monthly_growth_rate = np.power(predicted_annual_growth_rate, 1 / 12)
    current_volume = agent.parameter['total_production'].value
    # calculated using a fixed month-on-month growth rate from the most recent production volume

    proj = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        proj[i] = current_volume * pow(monthly_growth_rate, i + 1)

    return proj


def unit_sale_price_projection(agent) -> np.ndarray:
    # Calculate the projected PET sale prices
    proj = np.zeros(agent.projection_time)
    current = agent.env.parameter['pet_price'].value
    proj.fill(current)
    return proj


def fossil_feedstock_price_projection(agent) -> np.ndarray:
    # Calculate the projected PET feedstock costs
    proj = np.zeros(agent.projection_time)
    current = agent.env.parameter['fossil_feedstock_price'].value
    proj.fill(current)
    return proj


def fossil_process_cost_projection(agent) -> np.ndarray:
    # Calculate the projected PET processing costs
    proj = np.zeros(agent.projection_time)
    current = agent.parameter['fossil_process_cost'].value
    proj.fill(current)
    return proj


def proportion_bio_projection(agent) -> np.ndarray:
    # projection of proportion of production from bio routes
    proj = np.zeros(agent.projection_time)
    distance_to_target = (agent.proportion_bio_target - agent.parameter['proportion_bio'].value)
    time_to_target = int(np.ceil(abs(distance_to_target) /
                                 agent.proportion_change_rate)
                         + agent.implementation_countdown)

    proj.fill(agent.proportion_bio_target)
    if time_to_target > 1:

        for i in range(agent.implementation_countdown):
            proj[i] = agent.parameter['proportion_bio'].value

        for i in range(agent.implementation_countdown, time_to_target - 1):
            j = i - agent.implementation_countdown
            try:
                if distance_to_target > 0:
                    proj[i] = (agent.parameter['proportion_bio'].value +
                               agent.proportion_change_rate * (j + 1))
                else:
                    proj[i] = (agent.parameter['proportion_bio'].value -
                               agent.proportion_change_rate * (j + 1))

            except IndexError:
                print('time to reach target bio proportion is longer than', agent.projection_time, 'months')
                print('behaviour in these conditions is undefined. aborting simulation')
                raise SystemExit(0)

    else:
        pass

    return proj


def bio_feedstock_price_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.env.parameter['bio_feedstock_price'].value
    return proj


def bio_process_cost_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.parameter['bio_process_cost'].value
    return proj


def emissions_projection(agent) -> np.ndarray:
    monthly_production_projection = agent.parameter['fossil_production'].projection / 12

    proj = monthly_production_projection * agent.emissions_rate

    return proj


def levies_payable_projection(agent) -> np.ndarray:
    proj = np.multiply(agent.parameter['emissions'].projection, agent.parameter['levy_rate'].projection)
    return proj


def gross_profit_projection(agent) -> np.ndarray:
    # calculate revenues and costs at each month
    monthly_production_projection = agent.parameter['total_production'].projection / 12

    sales_projection = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        sales_projection[i] = min(agent.parameter['demand'].projection[i],
                                  agent.parameter['total_production'].projection[i])

    revenue_projection = np.multiply(sales_projection, agent.parameter['unit_sale_price'].projection)

    fossil_cost_projection = np.add(
        np.multiply(agent.parameter['fossil_feedstock_consumption'].projection,
                    agent.parameter['fossil_feedstock_price'].projection),
        np.multiply(monthly_production_projection,
                    np.multiply(agent.parameter['fossil_process_cost'].projection,
                                np.subtract(np.ones(agent.projection_time),
                                            agent.parameter['proportion_bio'].projection)
                                )
                    )
    )
    bio_cost_projection = np.add(
        np.multiply(agent.parameter['bio_feedstock_consumption'].projection,
                    agent.parameter['bio_feedstock_price'].projection),
        np.multiply(monthly_production_projection,
                    np.multiply(agent.parameter['bio_process_cost'].projection,
                                agent.parameter['proportion_bio'].projection)
                    )
    )
    capacity_maintenance_projection = (np.add(agent.parameter['fossil_capacity'].projection,
                                              agent.parameter['bio_capacity'].projection) *
                                       agent.capacity_maintenance_cost)

    total_cost_projection = np.add(
        np.add(fossil_cost_projection, bio_cost_projection),
        np.add(agent.parameter['levies_payable'].projection, capacity_maintenance_projection))

    proj = np.subtract(revenue_projection, total_cost_projection)
    return proj


def tax_rate_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.tax_rate
    return proj


def levy_rate_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    if not agent.env.levy_rate_changing:
        proj.fill(agent.env.parameter['levy_rate'].value)
    else:
        proj.fill(agent.env.future_levy_rate)
        for i in range(agent.env.time_to_levy_change):
            proj[i] = agent.env.parameter['levy_rate'].value
    return proj


def tax_payable_projection(agent) -> np.ndarray:
    proj = np.multiply(agent.parameter['gross_profit'].projection, agent.tax_rate_projection)
    return proj


def net_profit_projection(agent) -> np.ndarray:
    proj = np.subtract(agent.parameter['gross_profit'].projection, agent.parameter['tax_payable'].projection)
    return proj


def profitability_projection(agent) -> np.ndarray:
    proj = np.divide(agent.parameter['net_profit'].projection,
                     agent.parameter['total_production'].projection / 12)
    return proj


def expansion_cost_projection(agent) -> np.ndarray:
    bio_expansion = np.zeros(agent.projection_time)
    fossil_expansion = np.zeros(agent.projection_time)
    if agent.month > 0:
        bio_expansion[0] = (agent.parameter['bio_capacity'].value -
                            agent.parameter['bio_capacity'].history[agent.month - 1])
        fossil_expansion[0] = (agent.parameter['fossil_capacity'].value -
                               agent.parameter['fossil_capacity'].history[agent.month - 1])
    elif agent.month == 0:
        bio_expansion[0] = 0
        fossil_expansion[0] = 0
    else:
        pass

    for i in range(1, agent.projection_time):
        bio_expansion[i] = (agent.parameter['bio_capacity'].projection[i] -
                            agent.parameter['bio_capacity'].projection[i - 1])
        fossil_expansion[i] = (agent.parameter['fossil_capacity'].projection[i] -
                               agent.parameter['fossil_capacity'].projection[i - 1])

    bio_expansion_cost = bio_expansion * agent.bio_capacity_cost
    fossil_expansion_cost = fossil_expansion * agent.fossil_capacity_cost

    proj = np.add(bio_expansion_cost, fossil_expansion_cost)

    return proj


# def bio_capacity_projection(agent) -> np.ndarray:
#     production_bio = np.multiply(agent.parameter['production_volume'].projection,
#                                  agent.parameter['proportion_bio'].projection)
#     proj = np.zeros(agent.projection_time)
#     proj[0] = max(agent.parameter['bio_capacity'].value, production_bio[0])
#     for i in range(1, agent.projection_time):
#         proj[i] = max(production_bio[i], proj[i - 1])
#     return proj
#
#
# def fossil_capacity_projection(agent) -> np.ndarray:
#     production_fossil = np.multiply(agent.parameter['production_volume'].projection,
#                                     np.subtract(np.ones(agent.projection_time),
#                                                 agent.parameter['proportion_bio'].projection))
#     proj = np.zeros(agent.projection_time)
#     proj[0] = max(agent.parameter['fossil_capacity'].value, production_fossil[0])
#     for i in range(1, agent.projection_time):
#         proj[i] = max(production_fossil[i], proj[i - 1])
#     return proj


def liquidity_projection(agent) -> np.ndarray:
    liq = np.zeros(agent.projection_time)
    liq.fill(agent.parameter['liquidity'].value)

    revenues = np.cumsum(agent.parameter['net_profit'].projection, dtype=np.float64)
    costs = np.cumsum(agent.parameter['expansion_cost'].projection, dtype=np.float64)
    profits = np.subtract(revenues, costs)
    proj = np.add(liq, profits)
    return proj


def profit_margin_projection(agent) -> np.ndarray:
    monthly_production_projection = agent.parameter['total_production'].projection / 12
    proj = np.divide(agent.parameter['net_profit'].projection, monthly_production_projection)
    return proj


def fossil_feedstock_consumption(agent) -> np.float64:
    production_fossil = agent.parameter['fossil_production'].value / 12
    val = np.float64(production_fossil * agent.fossil_resource_ratio)
    return val


def fossil_feedstock_consumption_projection(agent) -> np.ndarray:
    monthly_production = agent.parameter['total_production'].projection / 12
    production_fossil = np.multiply(monthly_production,
                                    np.subtract(np.ones(agent.projection_time),
                                                agent.parameter['proportion_bio'].projection))
    proj = np.multiply(production_fossil, agent.fossil_resource_ratio)
    return proj


def bio_feedstock_consumption(agent) -> np.float64:
    production_bio = agent.parameter['bio_production'].value / 12
    val = np.float64(production_bio * agent.bio_resource_ratio)
    return val


def bio_feedstock_consumption_projection(agent) -> np.ndarray:
    production_bio = agent.parameter['bio_production'].projection / 12
    proj = np.multiply(production_bio, agent.bio_resource_ratio)
    return proj


"""new methods for change of Manufacturer structure to facilitate multi-objective multi-variate optimisation"""


def fossil_capacity_alt(agent) -> np.float64:
    current = agent.parameter['fossil_capacity'].value
    target = agent.fossil_capacity_target
    max_change = agent.build_rate
    distance_to_travel = current - target

    if agent.fossil_building:
        if abs(distance_to_travel) < max_change:
            val = target
        elif target < current:
            val = current - max_change
        else:
            val = current + max_change

    else:
        val = current

    return val


def bio_capacity_alt(agent) -> np.float64:
    current = agent.parameter['bio_capacity'].value
    target = agent.bio_capacity_target
    max_change = agent.build_rate
    distance_to_travel = current - target

    if agent.fossil_building:
        if abs(distance_to_travel) < max_change:
            val = target
        elif target < current:
            val = current - max_change
        else:
            val = current + max_change

    else:
        val = current

    return val


def production_scenario(production, agent):
    assert isinstance(production, np.ndarray)
    assert len(production) == 2
    sandbox = copy.deepcopy(agent)
    sandbox.parameter['fossil_production'].value = production[0]
    sandbox.parameter['bio_production'].value = production[1]

    update_start = max(sandbox.keys.index('fossil_production'),
                       sandbox.keys.index('bio_production'))

    for i in range(update_start, len(sandbox.keys)):
        key = sandbox.keys[i]
        sandbox.parameter[key].update()

    utility = -1 * sandbox.parameter['profitability'].value

    return utility


def fossil_production(agent) -> np.float64:
    current_fossil = agent.parameter['fossil_production'].value
    capacity_fossil = agent.parameter['fossil_capacity'].value

    current_bio = agent.parameter['fossil_production'].value
    capacity_bio = agent.parameter['fossil_capacity'].value

    optimum = optimize.minimize(production_scenario, np.ndarray([current_fossil, current_bio]), args=(agent),
                                method='l-bfgs-b', bounds=Bounds([0.0, 0.0], [capacity_fossil, capacity_bio]))

    output = optimum.x

    fossil_val = output[0]
    agent.parameter['bio_production'] = output[1]

    return fossil_val


def fossil_production_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    print('WARNING: Function parameter.fossil_production_projection is unfinished')
    run_check()

    return proj


def bio_production(agent) -> np.float64:
    return agent.parameter['bio_production'].value


def bio_production_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    print('WARNING: Function parameter.bio_production_projection is unfinished')
    run_check()

    return proj


def total_production(agent) -> np.float64:
    return agent.parameter['fossil_production'].value + agent.parameter['bio_production'].value


def total_production_projection(agent) -> np.ndarray:
    proj = np.add(agent.parameter['fossil_production'].projection,
                  agent.parameter['bio_production'].projection)
    return proj


def bio_capacity_max_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.parameter['bio_capacity_max'].value
    # keeps the maximum constant
    return proj


def fossil_capacity_max_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.parameter['fossil_capacity_max'].value
    return proj


def bio_capacity_projection(agent) -> np.ndarray:
    print('WARNING: Function parameter.bio_capacity_projection is unfinished')
    run_check()
    proj = np.zeros(agent.projection_time)
    return proj


def fossil_capacity_projection(agent) -> np.ndarray:
    print('WARNING: Function parameter.fossil_capacity_projection is unfinished')
    run_check()
    proj = np.zeros(agent.projection_time)
    return proj
