"""This file defines simulation parameters for the first build of the model"""
import agent as ag
from agent import Environment
from regulator import Regulator
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import copy
from parameter import Parameter
from parameter import Environment_Variable
import parameter as par
from datetime import datetime
from feed_supplier import Supplier


def simulate(months, table=False, plot=False):
    # create agents and specify their parameters
    month = int(0)
    initial_production_volume = np.float64(1000)

    # the dictionary of environment variables (see parameter.py) to pass to the Environment object
    env_variables = {
        'pet_price': Environment_Variable(par.pet_price, months, init=np.float64(5)),
        'fossil_feedstock_price': Environment_Variable(par.fossil_feedstock_price, months, init=np.float64(2)),
        'bio_feedstock_price': Environment_Variable(par.bio_feedstock_price, months, init=np.float64(2)),
        'levy_rate': Environment_Variable(par.levy_rate, months, init=np.float64(0.2)),
        'demand': Environment_Variable(par.demand, months, init=np.float64(1000))
    }

    env_keys = list(env_variables.keys())

    env_aggregates = {
        'fossil_feedstock_consumption': Environment_Variable(par.blank, months),
        'bio_feedstock_consumption': Environment_Variable(par.blank, months),
        'emissions': Environment_Variable(par.blank, months)
    }

    env_aggregates_keys = list(env_aggregates.keys())

    environment: Environment = ag.Environment(env_variables, env_aggregates)

    # dictionary of all variables in the order in which they should be computed
    # parameters from the environment that need to be projected by the agent use par.blank for the fun argument
    # similarly for parameters calculated by the agent but which are not projected
    manufacturer1_parameters = {
        'unit_sale_price': Parameter(par.blank, par.unit_sale_price_projection, months),
        'fossil_feedstock_price': Parameter(par.blank, par.fossil_feedstock_price_projection, months,
                                            init=np.float64(2)),
        'bio_feedstock_price': Parameter(par.blank, par.bio_feedstock_price_projection, months),
        'demand': Parameter(par.blank, par.demand_projection, months),

        # 'production_volume': Parameter(par.production_volume, par.production_volume_projection, months,
        #                                init=initial_production_volume),

        'fossil_process_cost': Parameter(par.fossil_process_cost, par.fossil_process_cost_projection, months,
                                         init=np.float64(1)),
        'bio_process_cost': Parameter(par.bio_process_cost, par.bio_process_cost_projection, months,
                                      init=np.float64(1.05)),

        'bio_capacity_max': Parameter(par.bio_capacity_max, par.bio_capacity_max_projection, months),
        'fossil_capacity_max': Parameter(par.fossil_capacity_max, par.fossil_capacity_max_projection, months),

        'fossil_capacity': Parameter(par.fossil_capacity_alt, par.fossil_capacity_projection, months,
                                     init=initial_production_volume),
        'bio_capacity': Parameter(par.bio_capacity_alt, par.bio_capacity_projection, months),
        'expansion_cost': Parameter(par.expansion_cost, par.expansion_cost_projection, months),

        'fossil_production': Parameter(par.fossil_production, par.fossil_production_projection, months,
                                       init=np.float64(1000)),
        'bio_production': Parameter(par.bio_production, par.bio_production_projection, months),
        'total_production': Parameter(par.total_production, par.total_production_projection, months,
                                      init=np.float64(1000)),

        'fossil_feedstock_consumption': Parameter(par.fossil_feedstock_consumption,
                                                  par.fossil_feedstock_consumption_projection, months),
        'bio_feedstock_consumption': Parameter(par.bio_feedstock_consumption,
                                               par.bio_feedstock_consumption_projection, months),

        'emissions': Parameter(par.emissions, par.emissions_projection, months),
        'levy_rate': Parameter(par.blank, par.levy_rate_projection, months, init=np.float64(0.2)),
        'levies_payable': Parameter(par.levies_payable, par.levies_payable_projection, months),

        'gross_profit': Parameter(par.gross_profit, par.gross_profit_projection, months),
        'tax_payable': Parameter(par.tax_payable, par.tax_payable_projection, months),
        'net_profit': Parameter(par.net_profit, par.net_profit_projection, months),

        'profitability': Parameter(par.profitability, par.profitability_projection, months),
        'liquidity': Parameter(par.liquidity, par.liquidity_projection, months, init=np.float64(5000)),
        'profit_margin': Parameter(par.profit_margin, par.profit_margin_projection, months)
    }

    manufacturer1 = ag.Manufacturer('PET Manufacturer 1', months, environment, manufacturer1_parameters)

    regulator = Regulator(name='Regulator', sim_time=months, env=environment, tax_rate=0.19, notice_period=24,
                          fraction=0.5, start_levy=0.2, ratio_jump=0.5, compliance_threshold=0.5, decade_jump=0.1)

    supplier = Supplier('supplier', months, environment, 2.0, 1000.0, 1000.0, 0.01, 0.5, 10, 0.02)

    manufacturers = [
        manufacturer1
    ]

    agents = manufacturers + [regulator, supplier]

    sim_start = datetime.now()
    # Run simulation for defined number of months
    while month < months:
        for key in env_keys:
            if month != 0:
                environment.parameter[key].update(environment)

            environment.parameter[key].record(month)

        # advance time counter in environment
        environment.month = month
        # advance time counter in each agent
        for agent in agents:
            agent.month = month

        # execute monthly routines on manufacturers
        for agent in manufacturers:
            agent.time_step_alt()

        environment.reset_aggregates()
        for key in env_aggregates_keys:
            for manufacturer in manufacturers:
                try:
                    environment.aggregate[key].value += manufacturer.parameter[key].value
                except KeyError:
                    pass

        environment.aggregate['bio_feedstock_consumption'].value += 500

        for key in env_aggregates_keys:
            environment.aggregate[key].record(month)

        supplier.iterate_supplier(False)

        regulator.iterate_regulator()

        # if the regulator rate has just changed then update it in the environment
        if environment.parameter['levy_rate'].value != regulator.levy_rate:
            environment.parameter['levy_rate'].value = regulator.levy_rate
            environment.time_to_levy_change = regulator.time_to_change()
            environment.levy_rate_changing = False

        # if a change in the levy rate is approaching, add this information to the environment
        if regulator.change_check():
            environment.levy_rate_changing = True
            environment.time_to_levy_change = regulator.time_to_change()
            environment.future_levy_rate = regulator.future_levy_rate
        else:
            pass

        month += 1

    sim_end = datetime.now()
    elapsed = sim_end - sim_start
    print('\n Simulation elapsed time:', elapsed)
    print('\n ============ \n FINAL STATE \n ============',
          '\n Regulation level:', regulator.level,
          '\n Levy rate:', environment.parameter['levy_rate'].value,
          '\n Bio capacity 1:', manufacturer1.parameter['bio_capacity'].value)
    print(' Target:', manufacturer1.bio_capacity_target)

    # data output & analysis
    t = np.arange(0, months, 1)

    if table:
        table = []
        for i in range(0, months):
            table.append([t[i],
                          environment.parameter['bio_feedstock_price'].history[i]])

        headers = ['Month', 'levy_rate']
        print(tabulate(table, headers))

    if plot:
        graph(manufacturer1.parameter['profitability'])

    return


def graph(parameter):
    assert isinstance(parameter, Parameter) or isinstance(parameter, Environment_Variable)
    y = parameter.history
    t = np.arange(0, len(y), 1)
    fig, ax1 = plt.subplots()
    ax1.plot(t, y)

    ax1.set_xlabel('Month')
    ax1.set_ylabel('profitability')

    fig.tight_layout()
    plt.show()
    return


def graph2(parameter1, parameter2):
    assert isinstance(parameter1, Parameter) or isinstance(parameter1, Environment_Variable)
    assert isinstance(parameter2, Parameter) or isinstance(parameter2, Environment_Variable)
    y1 = parameter1.history
    t1 = np.arange(0, len(y1), 1)
    y2 = parameter2.history
    t2 = np.arange(0, len(y2), 1)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (months)')
    ax1.set_ylabel('', color=color)
    ax1.plot(t1, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('', color=color)  # we already handled the x-label with ax1
    ax2.plot(t2, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return
