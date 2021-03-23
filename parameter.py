import numpy as np
import agent as ag


class Parameter(object):
    # an object class with a current value, a record of all past values, and a variable to hold a projection
    # only compatible with data types which can be parsed as a float
    # argument FUN is a function taking a single AGENT object as an argument and returns the next value of the variable
    def __init__(self, fun, project, sim_length, projection_time=120, init=np.float64(0)):
        assert type(sim_length) == int
        assert sim_length > 0
        assert type(projection_time) == int
        assert projection_time > 0

        self.fun = fun
        self.project = project

        self.value = np.float64(init)

        self.projection = np.zeros(projection_time)
        self.history = np.zeros(sim_length)

        return

    def update(self, actor):
        # calls the defined update function to calculate the next value of the variable
        assert isinstance(actor, ag.Agent)
        self.value = self.fun(actor)
        return

    def record(self, time):
        # writes the current value of the parameter to a chosen element of the record array
        self.history[time] = self.value
        return

    def forecast(self, actor):
        assert isinstance(actor, ag.Agent)
        self.projection = self.project(actor)
        return


def production_volume(agent) -> np.float64:
    volume = agent.parameter['production_volume'].value
    month = agent.month

    # production volume is defined by growth rates

    growth_rate = 1.02  # YoY growth rate for the second simulation period, expressed as a ratio
    growth_rate_monthly = np.power(growth_rate, 1 / 12)  # annual growth rate changed to month-on-month

    if month == 0:
        val = volume

    else:
        val = volume * growth_rate_monthly

    return val


def unit_sale_price(agent) -> np.float64:
    # unit sale price is given by a normal distribution
    mean = np.float64(4.5)
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
    # process cost is given by a normal distribution around a mean
    mean = np.float64(1)
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


def levy_rate(agent) -> np.float64:
    return agent.parameter['levy_rate'].value


def emissions(agent) -> np.float64:
    fossil_production = agent.parameter['production_volume'].value / 12 * (1 - agent.parameter['proportion_bio'].value)
    val = fossil_production * agent.emissions_rate
    return val


def levies_payable(agent) -> np.float64:
    """This will calculate the levies payable on production/consumption/emission, once they are defined"""
    val = agent.parameter['levy_rate'].value * agent.parameter['emissions'].value
    return val


def gross_profit(agent) -> np.float64:
    production_in_month = agent.parameter['production_volume'].value / 12
    revenue = production_in_month * agent.parameter['unit_sale_price'].value
    costs = (
            production_in_month *
            (
                (1 - agent.parameter['proportion_bio'].value) *
                (agent.parameter['unit_feedstock_cost'].value + agent.parameter['unit_process_cost'].value) +

                agent.parameter['proportion_bio'].value *
                (agent.parameter['bio_feedstock_cost'].value + agent.parameter['bio_process_cost'].value)
            )
            + agent.parameter['levies_payable'].value)
    val = revenue - costs
    return val


def tax_payable(agent) -> np.float64:
    val = agent.parameter['gross_profit'].value * agent.tax_rate
    return val


def net_profit(agent) -> np.float64:
    val = agent.parameter['gross_profit'].value - agent.parameter['tax_payable'].value
    return val


def profitability(agent) -> np.float64:
    val = agent.parameter['net_profit'].value / (agent.parameter['production_volume'].value / 12)
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


def bio_capacity(agent) -> np.float64:
    # only ever increases.
    val = np.float64()
    if agent.month > 0:
        prev = agent.parameter['bio_capacity'].history[agent.month - 1]
        now = agent.parameter['production_volume'].value * agent.parameter['proportion_bio'].value
        val = max(now, prev)
    elif agent.month == 0:
        val = 0
    else:
        pass
    return val


def fossil_capacity(agent) -> np.float64:
    val = np.float64()
    if agent.month > 0:
        prev = agent.parameter['fossil_capacity'].history[agent.month - 1]
        now = agent.parameter['production_volume'].value * (1 - agent.parameter['proportion_bio'].value)
        val = max(now, prev)
    elif agent.month == 0:
        val = 1000
    else:
        pass
    return val


def liquidity(agent) -> np.float64:
    val = (agent.parameter['liquidity'].value +
           agent.parameter['net_profit'].value -
           agent.parameter['expansion_cost'].value)
    return val


def profit_margin(agent) -> np.float64:
    production_in_month = agent.parameter['production_volume'].value / 12
    val = agent.parameter['net_profit'].value / (production_in_month * agent.parameter['unit_sale_price'].value)
    return val


# endregion

# region -- projection functions

def production_volume_projection(agent) -> np.ndarray:
    # calculates the projected (annualised) PET production volume for each month,
    # recording it to self.production_projection
    predicted_annual_growth_rate = 1.02
    monthly_growth_rate = np.power(predicted_annual_growth_rate, 1 / 12)
    current_volume = agent.parameter['production_volume'].value
    # calculated using a fixed month-on-month growth rate from the most recent production volume

    proj = np.empty(agent.projection_time)
    for i in range(agent.projection_time):
        proj[i] = current_volume * pow(monthly_growth_rate, i + 1)

    return proj


def unit_sale_price_projection(agent) -> np.ndarray:
    # Calculate the projected PET sale prices
    proj = np.zeros(agent.projection_time)
    proj.fill(4.5)  # fixed value
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
    time_to_target = int(np.ceil((agent.proportion_bio_target - agent.parameter['proportion_bio'].value) /
                                 agent.proportion_change_rate)
                         + agent.implementation_countdown)
    proj.fill(agent.proportion_bio_target)

    if time_to_target > 1:

        for i in range(agent.implementation_countdown):
            proj[i] = agent.parameter['proportion_bio'].value

        for i in range(agent.implementation_countdown, time_to_target - 1):
            try:
                proj[i] = (agent.parameter['proportion_bio'].value +
                           agent.proportion_change_rate * (i + 1))
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
    monthly_production_projection = agent.parameter['production_volume'].projection / 12

    proj = np.multiply(
        monthly_production_projection, np.subtract(
            np.ones(agent.projection_time), agent.parameter['proportion_bio'].projection)
    ) * agent.emissions_rate

    return proj


def levies_payable_projection(agent) -> np.ndarray:
    proj = np.multiply(agent.parameter['emissions'].projection, agent.parameter['levy_rate'].projection)
    return proj


def gross_profit_projection(agent) -> np.ndarray:
    # calculate revenues and costs at each month
    monthly_production_projection = agent.parameter['production_volume'].projection / 12

    revenue_projection = np.multiply(monthly_production_projection, agent.parameter['unit_sale_price'].projection)

    fossil_cost_projection = np.multiply(
        np.add(np.multiply(monthly_production_projection, agent.parameter['unit_feedstock_cost'].projection),
               np.multiply(monthly_production_projection, agent.parameter['unit_process_cost'].projection)),
        np.subtract(np.ones(agent.projection_time), agent.parameter['proportion_bio'].projection))
    bio_cost_projection = np.multiply(
        np.add(np.multiply(monthly_production_projection, agent.parameter['bio_feedstock_cost'].projection),
               np.multiply(monthly_production_projection, agent.parameter['bio_process_cost'].projection)),
        agent.parameter['proportion_bio'].projection)

    total_cost_projection = np.add(
        np.add(fossil_cost_projection, bio_cost_projection),
        agent.parameter['levies_payable'].projection)

    proj = np.subtract(revenue_projection, total_cost_projection)
    return proj


def tax_rate_projection(agent) -> np.ndarray:
    proj = np.ones(agent.projection_time) * agent.tax_rate
    return proj


def levy_rate_projection(agent) -> np.ndarray:
    proj = np.zeros(agent.projection_time)
    if not agent.levy_rate_changing:
        proj.fill(agent.parameter['levy_rate'].value)
    else:
        proj.fill(agent.future_levy_rate)
        for i in range(agent.time_to_levy_change):
            proj[i] = agent.parameter['levy_rate'].value
    return proj


def tax_payable_projection(agent) -> np.ndarray:
    proj = np.multiply(agent.parameter['gross_profit'].projection, agent.tax_rate_projection)
    return proj


def net_profit_projection(agent) -> np.ndarray:
    proj = np.subtract(agent.parameter['gross_profit'].projection, agent.parameter['tax_payable'].projection)
    return proj


def profitability_projection(agent) -> np.ndarray:
    proj = np.divide(agent.parameter['net_profit'].projection,
                     agent.parameter['production_volume'].projection / 12)
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


def bio_capacity_projection(agent) -> np.ndarray:
    bio_production = np.multiply(agent.parameter['production_volume'].projection,
                                 agent.parameter['proportion_bio'].projection)
    proj = np.zeros(agent.projection_time)
    proj[0] = max(agent.parameter['bio_capacity'].value, bio_production[0])
    for i in range(1, agent.projection_time):
        proj[i] = max(bio_production[i], proj[i - 1])
    return proj


def fossil_capacity_projection(agent) -> np.ndarray:
    fossil_production = np.multiply(agent.parameter['production_volume'].projection,
                                    np.subtract(np.ones(agent.projection_time), agent.parameter['proportion_bio'].projection))
    proj = np.zeros(agent.projection_time)
    proj[0] = max(agent.parameter['fossil_capacity'].value, fossil_production[0])
    for i in range(1, agent.projection_time):
        proj[i] = max(fossil_production[i], proj[i - 1])
    return proj


def liquidity_projection(agent) -> np.ndarray:
    liq = np.zeros(agent.projection_time)
    liq.fill(agent.parameter['liquidity'].value)

    revenues = np.cumsum(agent.parameter['net_profit'].projection, dtype=np.float64)
    costs = np.cumsum(agent.parameter['expansion_cost'].projection, dtype=np.float64)
    profits = np.subtract(revenues, costs)
    proj = np.add(liq, profits)
    return proj


def profit_margin_projection(agent) -> np.ndarray:
    monthly_production_projection = agent.parameter['production_volume'].projection / 12
    proj = np.divide(agent.parameter['net_profit'].projection, monthly_production_projection)
    return proj


# endregion
