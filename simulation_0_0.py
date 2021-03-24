"""This file defines simulation parameters for the first build of the model"""
import agent as ag
from regulator import Regulator
from regulator import Policy
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import copy
from parameter import Parameter
from parameter import Environment_Variable
import parameter as par


def simulate(months, table=False, plot=False):
    # create agents and specify their parameters
    month = int(0)
    initial_production_volume = np.float64(1000)

    environment = {
        'pet_price': Environment_Variable(par.pet_price, months, init=np.float64(4.5))
    }

    # dictionary of all variables in the order in which they should be computed
    manufacturer1_parameters = {
        'production_volume': Parameter(par.production_volume, par.production_volume_projection, months,
                                       init=initial_production_volume),
        'unit_sale_price': Parameter(par.unit_sale_price, par.unit_sale_price_projection, months,
                                     init=np.float64(4.5)),
        'unit_feedstock_cost': Parameter(par.unit_feedstock_cost, par.unit_feedstock_cost_projection, months,
                                         init=np.float64(2)),
        'unit_process_cost': Parameter(par.unit_process_cost, par.unit_process_cost_projection, months,
                                       init=np.float64(1)),
        'bio_feedstock_cost': Parameter(par.bio_feedstock_cost, par.bio_feedstock_cost_projection, months,
                                        init=np.float64(2)),
        'bio_process_cost': Parameter(par.bio_process_cost, par.bio_process_cost_projection, months,
                                      init=np.float64(1.05)),

        'proportion_bio': Parameter(par.proportion_bio, par.proportion_bio_projection, months),

        'bio_capacity': Parameter(par.bio_capacity, par.bio_capacity_projection, months),
        'fossil_capacity': Parameter(par.fossil_capacity, par.fossil_capacity_projection, months,
                                     init=initial_production_volume),
        'expansion_cost': Parameter(par.expansion_cost, par.expansion_cost_projection, months),

        'emissions': Parameter(par.emissions, par.emissions_projection, months),
        'levy_rate': Parameter(par.levy_rate, par.levy_rate_projection, months, init=np.float64(0.2)),
        'levies_payable': Parameter(par.levies_payable, par.levies_payable_projection, months),

        'gross_profit': Parameter(par.gross_profit, par.gross_profit_projection, months),
        'tax_payable': Parameter(par.tax_payable, par.tax_payable_projection, months),
        'net_profit': Parameter(par.net_profit, par.net_profit_projection, months),

        'profitability': Parameter(par.profitability, par.profitability_projection, months),
        'liquidity': Parameter(par.liquidity, par.liquidity_projection, months, init=np.float64(5000)),
        'profit_margin': Parameter(par.profit_margin, par.profit_margin_projection, months)
    }

    manufacturer1 = ag.Manufacturer('PET Manufacturer', manufacturer1_parameters, months, environment)

    policy = Policy()
    policy.add_level([1900, 0.19, 0.2])
    policy.add_level([2000, 0.19, 0.225])
    policy.add_level([2100, 0.19, 0.25])
    policy.add_level([2200, 0.19, 0.275])

    notice_period = int(18)

    regulator = Regulator('Regulator', months, notice_period, policy, environment)

    agents = [
        manufacturer1,
        regulator
    ]

    # Run simulation for defined number of months
    while month < months:
        # advance time counter in each agent
        for agent in agents:
            agent.month = month

        # execute standard monthly routines
        manufacturer1.time_step()
        regulator.iterate_regulator(manufacturer1.parameter['emissions'].value)

        # if the regulator rate has just changed (resulting in mismatch between agents) then update it
        if manufacturer1.parameter['levy_rate'].value != regulator.levy_rate:
            manufacturer1.parameter['levy_rate'].value = regulator.levy_rate
            manufacturer1.time_to_levy_change = copy.deepcopy(regulator.time_to_change)
            manufacturer1.levy_rate_changing = False

        # if a change in the levy rate is approaching, tell the pet_manufacturer
        if regulator.changing:
            manufacturer1.levy_rate_changing = True
            manufacturer1.time_to_levy_change = copy.deepcopy(regulator.time_to_change)
            manufacturer1.future_levy_rate = regulator.pol_table[regulator.level + 1][2]
        else:
            pass

        environment['pet_price'].update(environment)
        environment['pet_price'].record(month)

        month += 1

    print(' ============ \n FINAL STATE \n ============',
          '\n Regulation level:', regulator.level,
          '\n Levy rate:', regulator.levy_rate,
          '\n Bio proportion', manufacturer1.parameter['proportion_bio'].value)

    # data output & analysis
    t = np.arange(0, months, 1)

    if table:
        table = []
        for i in range(0, months):
            table.append([t[i],
                          manufacturer1.parameter['profitability'].history[i]])

        headers = ['Month', 'Profitability']
        print(tabulate(table, headers))

    if plot:
        y = environment['pet_price'].history
        x = t
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('')

        fig.tight_layout()
        plt.show()

    return
