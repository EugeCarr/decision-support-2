import numpy as np
import agent as ag
import copy
# from agent import run_check
from scipy import optimize
from scipy.optimize import Bounds


class Environment_Variable(object):
    # shared variables, such as market prices and aggregated demand, are stored as environment variables
    def __init__(self, fun, sim_length, init=np.float64(0)):
        assert type(sim_length) == int
        assert sim_length > 0

        self.month = int(0)

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
    # std_dev = 0.01
    # deviation = np.float64(np.random.normal(0, std_dev, None))
    # val += deviation

    # val = env.parameter['pet_price'].history[0]
    return val


def fossil_feedstock_price(env) -> np.float64:
    # price is a random walk from the initial value
    val = env.parameter['fossil_feedstock_price'].value
    # std_dev = 0.01
    # deviation = np.float64(np.random.normal(0, std_dev, None))
    # val += deviation

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
    growth_rate = 1.05  # YoY growth rate, expressed as a ratio
    growth_rate_monthly = np.power(growth_rate, 1 / 12)  # annual growth rate changed to month-on-month
    val = current * growth_rate_monthly
    # std_dev = 1
    # random = np.float64(np.random.normal(0, std_dev, None))
    # val += random

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

# production volume is defined by growth rates

def demand_projection(agent) -> np.ndarray:
    current = agent.env.parameter['demand'].value
    growth_rate = 1.05  # YoY growth rate for the simulation period, expressed as a ratio
    growth_rate_monthly = np.power(growth_rate, 1 / 12)  # annual growth rate changed to month-on-month

    proj = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        proj[i] = current * pow(growth_rate_monthly, i + 1)

    return proj


# def proportion_bio(agent) -> np.float64:
#     # monthly change in bio proportion is either the amount to reach the target value, or else the maximum change
#     val = np.float64(agent.parameter['proportion_bio'].value)
#     if agent.implementation_countdown == 0 and agent.parameter['proportion_bio'].value != agent.proportion_bio_target:
#         agent.under_construction = True
#         distance_from_target = agent.proportion_bio_target - agent.parameter['proportion_bio'].value
#         if abs(distance_from_target) < agent.proportion_change_rate:
#             val = agent.proportion_bio_target
#         elif distance_from_target > 0:
#             val += agent.proportion_change_rate
#         elif distance_from_target < 0:
#             val -= agent.proportion_change_rate
#         else:
#             pass
#     else:
#         agent.under_construction = False
#
#     return val


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


def fossil_capacity_max(agent) -> np.float64:
    baseline_capacity = agent.parameter['total_production'].history[0]
    if agent.month == 0:
        baseline_capacity = agent.parameter['total_production'].value
    max_cap = baseline_capacity * 2
    # defines an arbitrary maximum capacity as the starting production in month 0 multiplied by two
    return max_cap


def fossil_capacity_alt(agent) -> np.float64:
    current = agent.parameter['fossil_capacity'].value
    target = agent.fossil_capacity_target
    max_change = agent.change_rate
    distance_to_travel = current - target

    if agent.fossil_building:
        if abs(distance_to_travel) < max_change:
            val = target
            agent.fossil_building = False
        elif target < current:
            val = current - max_change
        else:
            val = current + max_change

    else:
        val = current

    return val


def fossil_capacity_alt2(agent) -> np.float64:
    current = agent.parameter['fossil_capacity'].value
    target = agent.fossil_capacity_target
    distance_to_target = target - current

    if agent.fossil_building:
        print('building fossil, current capacity:', current, 'target:', target)
        agent.fossil_building_month += 1

        expansion_change = curve_change_capacity(agent.fossil_building_month, agent)
        if abs(distance_to_target) < expansion_change:
            print('Change finished, current fossil cap:', current, 'future capacity:', target)
            val = target
            agent.fossil_building = False
            agent.fossil_building_month = 0
        elif abs(distance_to_target) > expansion_change and distance_to_target > 0:
            val = current + expansion_change
        elif abs(distance_to_target) > expansion_change and distance_to_target < 0:
            val = current - expansion_change

    else:
        val = current
        agent.fossil_building_month = 0

    return val


def bio_capacity_max(agent) -> np.float64:
    baseline_capacity = agent.parameter['total_production'].history[0]
    if agent.month == 0:
        baseline_capacity = agent.parameter['total_production'].value
    max_cap = baseline_capacity * 2
    # defines an arbitrary maximum capacity as the starting production in month 0 multiplied by two
    return max_cap


def bio_capacity_alt(agent) -> np.float64:
    current = agent.parameter['bio_capacity'].value
    target = agent.bio_capacity_target
    max_change = agent.change_rate
    distance_to_travel = current - target

    if agent.bio_building:
        if abs(distance_to_travel) < max_change:
            val = target
            agent.bio_building = False
        elif target < current:
            val = current - max_change
        else:
            val = current + max_change

    else:
        val = current

    return val


def bio_capacity_alt2(agent) -> np.float64:
    current = agent.parameter['bio_capacity'].value
    target = agent.bio_capacity_target
    distance_to_travel = target - current

    if agent.bio_building:
        print('building')
        agent.bio_building_month += 1
        print('building bio, current capacity:', current, 'target:', target)

        expansion_change = curve_change_capacity(agent.bio_building_month, agent)
        if abs(distance_to_travel) < expansion_change:
            print('Change finished, current bio cap:', current, 'future capacity:', target)
            val = target
            agent.bio_building = False
            agent.bio_building_month = 0

        elif abs(distance_to_travel) > expansion_change and distance_to_travel > 0:
            val = current + expansion_change

        elif abs(distance_to_travel) > expansion_change and distance_to_travel < 0:
            val = current - expansion_change

    else:
        val = current
        agent.bio_building_month = 0

    return val


def curve_change_capacity(month, company):
    assert type(month) == int and month > 0, ("year input", month, "for capacity build incorrect, must be 0 and int")
    assert isinstance(company, ag.Manufacturer), ("input", company, "is type:", type(company))

    baseline_capacity = company.parameter['fossil_capacity'].history[0]
    build_speed = company.build_speed
    # print(company.time_to_build * 12)
    # current_added = build_speed * baseline_capacity * np.power((month / company.sim_time),
    #                                                            (1 / company.capacity_root_coefficient))
    current_added = build_speed * baseline_capacity * np.power((month / int(company.time_to_build * 12)),
                                                               (1 / company.capacity_root_coefficient))
    # last_month_added = build_speed * baseline_capacity * np.power(((month - 1) / company.sim_time),
    #                                                               (1 / company.capacity_root_coefficient))
    last_month_added = build_speed * baseline_capacity * np.power(((month - 1) / int(company.time_to_build * 12)),
                                                                  (1 / company.capacity_root_coefficient))
    expansion_to_add = current_added - last_month_added

    return expansion_to_add


def expansion_cost(agent) -> np.float64:
    val = np.float64()
    if agent.month > 0:
        bio_change = abs(agent.parameter['bio_capacity'].value -
                         agent.parameter['bio_capacity'].history[agent.month - 1])
        bio_cost = agent.bio_capacity_cost * bio_change

        fossil_change = abs(agent.parameter['fossil_capacity'].value -
                            agent.parameter['fossil_capacity'].history[agent.month - 1])
        fossil_cost = agent.fossil_capacity_cost * fossil_change

        val = np.float64(bio_cost + fossil_cost)
    elif agent.month == 0:
        val = 0

    return val


def fossil_production(agent) -> np.float64:
    current_fossil = agent.parameter['fossil_production'].value
    capacity_fossil = agent.parameter['fossil_capacity'].value

    current_bio = agent.parameter['bio_production'].value
    capacity_bio = agent.parameter['bio_capacity'].value

    x0 = np.array([current_fossil, current_bio])

    if capacity_bio > 0:
        optimum = optimize.minimize(production_scenario, x0, args=(agent,),
                                    method='l-bfgs-b', bounds=Bounds([capacity_fossil * 0.7, capacity_bio * 0.7],
                                                                     [capacity_fossil, capacity_bio]))

        output = optimum.x

        fossil_val = np.round(output[0])
        agent.parameter['bio_production'].value = np.round(output[1])

    else:
        optimum = optimize.minimize_scalar(production_scenario_fossil, bounds=(0.0, capacity_fossil),
                                           args=(agent,), method='bounded')

        output = optimum.x
        fossil_val = np.round(output)
        agent.parameter['bio_production'].value = 0.0

    return fossil_val


def bio_production(agent) -> np.float64:
    return agent.parameter['bio_production'].value


def total_production(agent) -> np.float64:
    return agent.parameter['fossil_production'].value + agent.parameter['bio_production'].value


def fossil_utilisation(agent) -> np.float64:
    return agent.parameter['fossil_production'].value / agent.parameter['fossil_capacity'].value


def bio_utilisation(agent) -> np.float64:
    return agent.parameter['bio_production'].value / agent.parameter['bio_capacity'].value


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


def fossil_feedstock_consumption(agent) -> np.float64:
    production_fossil = agent.parameter['fossil_production'].value / 12
    val = np.float64(production_fossil * agent.fossil_resource_ratio)
    return val


def bio_feedstock_consumption(agent) -> np.float64:
    production_bio = agent.parameter['bio_production'].value / 12
    val = np.float64(production_bio * agent.bio_resource_ratio)
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
    sellable_amount = agent.env.parameter['demand'].value / 12
    product_sold = min(sellable_amount, production_in_month)
    revenue = product_sold * agent.env.parameter['pet_price'].value

    # costs calculation
    costs = 0
    costs += (agent.parameter['fossil_feedstock_consumption'].value *
              agent.env.parameter['fossil_feedstock_price'].value)

    costs += (agent.parameter['bio_feedstock_consumption'].value *
              agent.env.parameter['bio_feedstock_price'].value)

    costs += (agent.parameter['fossil_production'].value / 12 *
              agent.parameter['fossil_process_cost'].value)

    costs += (agent.parameter['bio_production'].value / 12 *
              agent.parameter['bio_process_cost'].value)

    costs += agent.parameter['levies_payable'].value

    costs += agent.capacity_maintenance_cost * (agent.parameter['fossil_capacity'].value +
                                                agent.parameter['bio_capacity'].value)

    val = revenue - costs
    return val


def tax_payable(agent) -> np.float64:
    val = agent.parameter['gross_profit'].value * agent.tax_rate
    return val


def net_profit(agent) -> np.float64:
    val = agent.parameter['gross_profit'].value - agent.parameter['tax_payable'].value
    return val


def profit_margin(agent) -> np.float64:
    production_in_month = agent.parameter['total_production'].value / 12
    val = agent.parameter['net_profit'].value / (production_in_month * agent.env.parameter['pet_price'].value)
    return val


def profitability(agent) -> np.float64:
    val = agent.parameter['net_profit'].value / (agent.parameter['total_production'].value / 12)
    return val


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


def production_scenario(production, agent):
    assert isinstance(production, np.ndarray)
    assert len(production) == 2
    sandbox = copy.deepcopy(agent)
    sandbox.parameter['fossil_production'].value = production[0]
    sandbox.parameter['bio_production'].value = production[1]

    update_start = max(sandbox.keys.index('fossil_production'),
                       sandbox.keys.index('bio_production')) + 1
    # starts from the highest index of bio and fossil production adds 1 for the starting index
    for i in range(update_start, len(sandbox.keys)):
        key = sandbox.keys[i]
        sandbox.parameter[key].update(sandbox)
    # the list of keys gives the order of parameter calculations
    # the rest of the parameters that depend on production rates are then calculated

    utility = -1 * sandbox.parameter['profitability'].value
    # the utility is minus the profitability at this point in time, with the input production parameters
    return utility


def production_scenario_fossil(production, agent):
    utility = production_scenario(np.array([production, 0.0]), agent)
    return utility


# def production_volume_projection(agent) -> np.ndarray:
#     # calculates the projected (annualised) PET production volume for each month,
#     # recording it to self.production_projection
#     predicted_annual_growth_rate = 1.02
#     monthly_growth_rate = np.power(predicted_annual_growth_rate, 1 / 12)
#     current_volume = agent.parameter['total_production'].value
#     # calculated using a fixed month-on-month growth rate from the most recent production volume
#
#     proj = np.empty(agent.projection_time)
#     for i in range(agent.projection_time):
#         proj[i] = current_volume * pow(monthly_growth_rate, i + 1)
#
#     return proj


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


# def proportion_bio_projection(agent) -> np.ndarray:
#     # projection of proportion of production from bio routes
#     proj = np.zeros(agent.projection_time)
#     distance_to_target = (agent.proportion_bio_target - agent.parameter['proportion_bio'].value)
#     time_to_target = int(np.ceil(abs(distance_to_target) /
#                                  agent.proportion_change_rate)
#                          + agent.implementation_countdown)
#
#     proj.fill(agent.proportion_bio_target)
#     if time_to_target > 1:
#
#         for i in range(agent.implementation_countdown):
#             proj[i] = agent.parameter['proportion_bio'].value
#
#         for i in range(agent.implementation_countdown, time_to_target - 1):
#             j = i - agent.implementation_countdown
#             try:
#                 if distance_to_target > 0:
#                     proj[i] = (agent.parameter['proportion_bio'].value +
#                                agent.proportion_change_rate * (j + 1))
#                 else:
#                     proj[i] = (agent.parameter['proportion_bio'].value -
#                                agent.proportion_change_rate * (j + 1))
#
#             except IndexError:
#                 print('time to reach target bio proportion is longer than', agent.projection_time, 'months')
#                 print('behaviour in these conditions is undefined. aborting simulation')
#                 raise SystemExit(0)
#
#     else:
#         pass
#
#     return proj


def bio_feedstock_price_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    current = agent.env.parameter['bio_feedstock_price'].value
    proj.fill(current)
    price_decline = np.ones(agent.projection_time)
    monthly_multiplier = np.power((1 - agent.env.ann_feed_price_decrease), 1 / 12)

    for i in range(len(price_decline)):
        index_multiplier = np.power(monthly_multiplier, i)
        price_decline[i] *= index_multiplier
    res = np.multiply(proj, price_decline)
    return res


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
    monthly_demand_projection = agent.parameter['demand'].projection / 12

    sales_projection = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        sales_projection[i] = min(monthly_demand_projection[i],
                                  monthly_production_projection[i])

    revenue_projection = np.multiply(sales_projection, agent.parameter['unit_sale_price'].projection)

    fossil_cost_projection = np.add(
        np.multiply(agent.parameter['fossil_feedstock_consumption'].projection,
                    agent.parameter['fossil_feedstock_price'].projection),
        np.multiply(agent.parameter['fossil_process_cost'].projection,
                    agent.parameter['fossil_production'].projection / 12)
    )

    bio_cost_projection = np.add(
        np.multiply(agent.parameter['bio_feedstock_consumption'].projection,
                    agent.parameter['bio_feedstock_price'].projection),
        np.multiply(agent.parameter['bio_process_cost'].projection,
                    agent.parameter['bio_production'].projection / 12)
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
    if min(agent.parameter['total_production'].projection) < 10:
        proj = np.empty(agent.projection_time)
        for i in range(agent.projection_time):
            if agent.parameter['total_production'].projection[i] > 0:
                proj[i] = (agent.parameter['net_profit'].projection[i] /
                           (agent.parameter['total_production'].projection[i] / 12))
            else:
                proj[i] = np.float64(0)
    else:
        proj = np.divide(agent.parameter['net_profit'].projection,
                         agent.parameter['total_production'].projection / 12)
    return proj


def expansion_cost_projection(agent) -> np.ndarray:
    bio_expansion = np.zeros(agent.projection_time)
    fossil_expansion = np.zeros(agent.projection_time)
    if agent.month > 0:
        bio_expansion[0] = abs(agent.parameter['bio_capacity'].value -
                               agent.parameter['bio_capacity'].history[agent.month - 1])
        fossil_expansion[0] = abs(agent.parameter['fossil_capacity'].value -
                                  agent.parameter['fossil_capacity'].history[agent.month - 1])
    elif agent.month == 0:
        bio_expansion[0] = 0
        fossil_expansion[0] = 0
    else:
        pass

    for i in range(1, agent.projection_time):
        bio_expansion[i] = abs(agent.parameter['bio_capacity'].projection[i] -
                               agent.parameter['bio_capacity'].projection[i - 1])
        fossil_expansion[i] = abs(agent.parameter['fossil_capacity'].projection[i] -
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


def fossil_feedstock_consumption_projection(agent) -> np.ndarray:
    production_fossil = agent.parameter['fossil_production'].projection / 12
    proj = production_fossil * agent.fossil_resource_ratio
    return proj


def bio_feedstock_consumption_projection(agent) -> np.ndarray:
    production_bio = agent.parameter['bio_production'].projection / 12
    proj = production_bio * agent.bio_resource_ratio
    return proj


def bio_capacity_projection_alt2(agent) -> np.ndarray:
    assert isinstance(agent, ag.Manufacturer)
    current = agent.parameter['bio_capacity'].value
    target = agent.bio_capacity_target
    distance_to_target = abs(current - target)

    if target == current:
        proj = np.ones(agent.projection_time) * current
    elif target > current:
        proj = np.zeros(agent.projection_time)
        baseline_capacity = agent.parameter['fossil_capacity'].history[0]

        build_speed = agent.build_speed

        # months_to_completion = int(
        #     np.ceil(agent.sim_time * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
        #                                       agent.capacity_root_coefficient)) + agent.design_time)
        months_to_completion = int(
            np.ceil(int(agent.time_to_build * 12) * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
                                                         agent.capacity_root_coefficient)) + agent.design_time)
        for i in range(agent.design_time):
            proj[i] = current
        for i in range(agent.design_time, min(agent.projection_time - 1, months_to_completion)):
            j = 1 + i - agent.design_time
            expansion_to_add = curve_change_capacity(j, agent)

            if expansion_to_add > target - proj[i - 1]:
                proj[i] = target
            else:
                proj[i] = proj[i - 1] + expansion_to_add

        if months_to_completion < agent.projection_time:
            for i in range(months_to_completion, agent.projection_time):
                proj[i] = target

    elif target < current:
        proj = np.zeros(agent.projection_time)
        baseline_capacity = agent.parameter['fossil_capacity'].history[0]

        build_speed = agent.build_speed

        # months_to_completion = int(
        #     np.ceil(agent.sim_time * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
        #                                       agent.capacity_root_coefficient)) + agent.design_time)
        months_to_completion = int(
            np.ceil(
                int(agent.time_to_build * 12) * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
                                                         agent.capacity_root_coefficient)) + agent.design_time)
        for i in range(agent.design_time):
            proj[i] = current
        for i in range(agent.design_time, min(agent.projection_time - 1, months_to_completion)):
            j = 1 + i - agent.design_time
            expansion_to_add = curve_change_capacity(j, agent)

            if expansion_to_add > abs(target - proj[i - 1]):
                proj[i] = target
            else:
                proj[i] = proj[i - 1] - expansion_to_add

        if months_to_completion < agent.projection_time:
            for i in range(months_to_completion, agent.projection_time):
                proj[i] = target

    return proj


def fossil_production_projection(agent) -> np.ndarray:
    # assumes target utilisation of capacity
    proj = agent.parameter['fossil_capacity'].projection * agent.fossil_utilisation_target
    return proj


def bio_production_projection(agent) -> np.ndarray:
    # assumes target utilisation
    proj = agent.parameter['bio_capacity'].projection * agent.bio_utilisation_target
    return proj


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
    current = agent.parameter['bio_capacity'].value
    target = agent.bio_capacity_target
    if target == current:
        proj = np.ones(agent.projection_time) * target

    else:
        max_change = agent.change_rate
        distance_to_travel = abs(current - target)
        months_to_completion = int(np.ceil(distance_to_travel / max_change) + agent.bio_build_countdown)
        proj = np.zeros(agent.projection_time)

        for i in range(agent.bio_build_countdown):  # delay if there is any remaining
            proj[i] = current
        for i in range(agent.bio_build_countdown, months_to_completion):  # ramp during building process
            j = i - agent.bio_build_countdown
            if target > current:
                proj[i] = current + max_change * j
            else:
                proj[i] = current - max_change * j
        for i in range(months_to_completion, agent.projection_time):  # finished state
            proj[i] = target

    return proj


def fossil_capacity_projection(agent) -> np.ndarray:
    current = agent.parameter['fossil_capacity'].value
    target = agent.fossil_capacity_target
    if target == current:
        proj = np.ones(agent.projection_time) * target

    else:
        max_change = agent.change_rate
        distance_to_travel = abs(current - target)
        months_to_completion = int(np.ceil(distance_to_travel / max_change) + agent.fossil_build_countdown)
        proj = np.zeros(agent.projection_time)

        for i in range(agent.fossil_build_countdown):  # delay if there is any remaining
            proj[i] = current
        for i in range(agent.fossil_build_countdown, months_to_completion):  # ramp during building process
            j = i - agent.fossil_build_countdown
            if target > current:
                proj[i] = current + max_change * j
            else:
                proj[i] = current - max_change * j
        for i in range(months_to_completion, agent.projection_time):  # finished state
            proj[i] = target

    return proj


def fossil_capacity_projection_alt2(agent) -> np.ndarray:
    assert isinstance(agent, ag.Manufacturer)
    current = agent.parameter['fossil_capacity'].value
    target = agent.fossil_capacity_target
    distance_to_target = abs(current - target)

    if target == current:
        proj = np.ones(agent.projection_time) * current
    elif target > current:
        proj = np.zeros(agent.projection_time)
        baseline_capacity = agent.parameter['fossil_capacity'].history[0]

        build_speed = agent.build_speed

        # months_to_completion = int(
        #     np.ceil(agent.sim_time * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
        #                                       agent.capacity_root_coefficient)) + agent.design_time)
        months_to_completion = int(
            np.ceil(
                int(agent.time_to_build * 12) * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
                                                         agent.capacity_root_coefficient)) + agent.design_time)
        for i in range(agent.design_time):
            proj[i] = current
        for i in range(agent.design_time, min(agent.projection_time - 1, months_to_completion)):
            j = 1 + i - agent.design_time
            expansion_to_add = curve_change_capacity(j, agent)

            if expansion_to_add > target - proj[i - 1]:
                proj[i] = target
            else:
                proj[i] = proj[i - 1] + expansion_to_add

        if months_to_completion < agent.projection_time:
            for i in range(months_to_completion, agent.projection_time):
                proj[i] = target

    elif target < current:
        proj = np.zeros(agent.projection_time)
        baseline_capacity = agent.parameter['fossil_capacity'].history[0]

        build_speed = agent.build_speed

        # months_to_completion = int(
        #     np.ceil(agent.sim_time * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
        #                                       agent.capacity_root_coefficient)) + agent.design_time)
        months_to_completion = int(
            np.ceil(
                int(agent.time_to_build * 12) * np.power((abs(distance_to_target) / (build_speed * baseline_capacity)),
                                                         agent.capacity_root_coefficient)) + agent.design_time)
        for i in range(agent.design_time):
            proj[i] = current
        for i in range(agent.design_time, min(agent.projection_time - 1, months_to_completion)):
            j = 1 + i - agent.design_time
            expansion_to_add = curve_change_capacity(j, agent)

            if expansion_to_add > abs(target - proj[i - 1]):
                proj[i] = target
            else:
                proj[i] = proj[i - 1] - expansion_to_add

        if months_to_completion < agent.projection_time:
            for i in range(months_to_completion, agent.projection_time):
                proj[i] = target

    return proj
