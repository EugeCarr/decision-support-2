"""This file defines simulation parameters for the first build of the model"""
import agent as ag
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
# import pandas as pd
import openpyxl


def simulate(months, table=False, plot=False, Excel_p=False):
    # def simulate(months, table=False, plot=False):
    # create agents and specify their parameters
    month = int(0)
    initial_production_volume = np.float64(1000)

    # the dictionary of environment variables (see parameter.py) to pass to the Environment object
    env_variables = {
        'pet_price': Environment_Variable(par.pet_price, months, init=np.float64(4.5)),
        'fossil_feedstock_price': Environment_Variable(par.fossil_feedstock_price, months, init=np.float64(2)),
        'bio_feedstock_price': Environment_Variable(par.bio_feedstock_price, months, init=np.float64(2)),
        'levy_rate': Environment_Variable(par.levy_rate, months, init=np.float64(0.2))
    }

    env_keys = list(env_variables.keys())

    env_aggregates = {
        'fossil_feedstock_consumption': Environment_Variable(par.blank, months),
        'bio_feedstock_consumption': Environment_Variable(par.blank, months),
        'emissions': Environment_Variable(par.blank, months)
    }

    env_aggregates_keys = list(env_aggregates.keys())

    environment = ag.Environment(env_variables, env_aggregates)

    # dictionary of all variables in the order in which they should be computed
    # parameters from the environment that need to be projected by the agent use par.blank for the fun argument
    # similarly for parameters calculated by the agent but which are not projected
    manufacturer1_parameters = {
        'unit_sale_price': Parameter(par.blank, par.unit_sale_price_projection, months),
        'fossil_feedstock_price': Parameter(par.blank, par.fossil_feedstock_price_projection, months,
                                            init=np.float64(2)),
        'bio_feedstock_price': Parameter(par.blank, par.bio_feedstock_price_projection, months),

        'production_volume': Parameter(par.production_volume, par.production_volume_projection, months,
                                       init=initial_production_volume),

        'fossil_process_cost': Parameter(par.fossil_process_cost, par.fossil_process_cost_projection, months,
                                         init=np.float64(1)),
        'bio_process_cost': Parameter(par.bio_process_cost, par.bio_process_cost_projection, months,
                                      init=np.float64(1.05)),

        'proportion_bio': Parameter(par.proportion_bio, par.proportion_bio_projection, months),

        'fossil_feedstock_consumption': Parameter(par.fossil_feedstock_consumption,
                                                  par.fossil_feedstock_consumption_projection, months),
        'bio_feedstock_consumption': Parameter(par.bio_feedstock_consumption,
                                               par.bio_feedstock_consumption_projection, months),

        'bio_capacity': Parameter(par.bio_capacity, par.bio_capacity_projection, months),
        'fossil_capacity': Parameter(par.fossil_capacity, par.fossil_capacity_projection, months,
                                     init=initial_production_volume),
        'expansion_cost': Parameter(par.expansion_cost, par.expansion_cost_projection, months),

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

    manufacturer2_parameters = copy.deepcopy(manufacturer1_parameters)

    manufacturer1 = ag.Manufacturer('PET Manufacturer 1', months, environment, manufacturer1_parameters)
    manufacturer2 = ag.Manufacturer('PET Manufacturer 2', months, environment, manufacturer2_parameters)

    regulator = Regulator(name='Regulator', sim_time=months, env=environment, tax_rate=0.19, fraction=0.7,
                          ratio_jump=0.5,
                          start_levy=0.2, decade_jump=3.0)

    supplier = Supplier('supplier', months, environment, 2.0)

    manufacturers = [
        manufacturer1,
        manufacturer2
    ]

    agents = manufacturers + [regulator, supplier]

    sim_start = datetime.now()
    # Run simulation for defined number of months
    while month < months:
        for key in env_keys:
            if month != 0:
                environment.parameter[key].update(environment)

            environment.parameter[key].record(month)

        # advance time counter in each agent
        for agent in agents:
            agent.month = month

        # execute monthly routines on manufacturers
        for agent in manufacturers:
            agent.time_step()
        regulator.iterate_regulator()
        supplier.iterate_supplier(False)

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
          '\n Bio proportion 1:', manufacturer1.parameter['proportion_bio'].value)

    # data output & analysis
    t = np.arange(0, months, 1)

    if table:
        table = []
        for i in range(0, months):
            table.append([t[i],
                          environment.parameter['levy_rate'].history[i]])

        headers = ['Month', 'levy_rate']
        print(tabulate(table, headers))

    if plot:
        # graph(manufacturer1.parameter['proportion_bio'])
        # graph(environment.parameter['bio_feedstock_price'])
        # graph(environment.parameter['levy_rate'])
        graph(manufacturer1.parameter['emissions'])

    if Excel_p:
        # bio_proportion_list = np.divide(manufacturer1.parameter['bio_production'].history,
        #                                 (manufacturer1.parameter['bio_production'].history + manufacturer1.parameter[
        #                                     'fossil_production'].history))

        # when the changes from rewrite_optimisation are merged in, a new parameter needs to be made for bio_proportion
        wb = openpyxl.load_workbook('Results from simulations.xlsx')
        print(type(wb))
        sheet = wb.create_sheet(title='First Try')
        print(sheet.title)
        date_time = datetime.now()

        cell_write(sheet, (1, 1), 'Simulation of decision support tool')

        cell_write(sheet, (3, 1), 'General information')
        cell_write(sheet, (4, 1), 'Date & time:')
        cell_write(sheet, (4, 2), date_time)
        cell_write(sheet, (5, 1), 'No. of Manufacturers')
        cell_write(sheet, (5, 2), len(manufacturers))
        # cell_write(sheet, (6, 1), 'Run time')
        # cell_write(sheet, (6, 2), elapsed)

        cell_write(sheet, (3, 4), 'Regulator Settings')
        cell_write(sheet, (4, 4), 'Tax rate')
        cell_write(sheet, (4, 5), regulator.tax_rate)
        cell_write(sheet, (5, 4), 'Emission fraction width')
        cell_write(sheet, (5, 5), regulator.fraction)
        cell_write(sheet, (6, 4), 'Emission ratio jump')
        cell_write(sheet, (6, 5), regulator.ratio_jump)
        cell_write(sheet, (7, 4), 'Initial levyrate')
        cell_write(sheet, (7, 5), regulator.start_levy)
        cell_write(sheet, (8, 4), 'Decade Change')
        cell_write(sheet, (8, 5), regulator.dec_jump)

        cell_write(sheet, (3, 6), 'Supplier Settings')
        cell_write(sheet, (4, 6), 'Initial price')
        cell_write(sheet, (4, 7), supplier.start_price)
        cell_write(sheet, (5, 6), 'Price elasticity')
        cell_write(sheet, (5, 7), supplier.price_elasticity)

        cell_write(sheet, (3, 8), 'Manufacturer 1 settings')
        cell_write(sheet, (4, 8), 'Staring liquidity')
        cell_write(sheet, (4, 9), manufacturer1.parameter['liquidity'].history[0])
        cell_write(sheet, (5, 8), 'Bio process cost')
        cell_write(sheet, (5, 9), manufacturer1.parameter['bio_process_cost'].history[0])
        cell_write(sheet, (7, 8), 'Fossil process cost')
        cell_write(sheet, (7, 9), manufacturer1.parameter['fossil_process_cost'].history[0])
        cell_write(sheet, (6, 8), 'Bio feedstock price')
        cell_write(sheet, (6, 9), environment.parameter['bio_feedstock_price'].history[0])
        print(environment.parameter['bio_feedstock_price'].history[0])
        cell_write(sheet, (8, 8), 'Fossil feedstock price')
        cell_write(sheet, (8, 9), environment.parameter['fossil_feedstock_price'].history[1])
        print(environment.parameter['fossil_feedstock_price'].history[0])
        cell_write(sheet, (9, 8), 'Starting Production')
        cell_write(sheet, (9, 9), manufacturer1.parameter['production_volume'].history[0])
        cell_write(sheet, (10, 8), 'Initial PET price')
        cell_write(sheet, (10, 9), environment.parameter['pet_price'].history[0])
        print(environment.parameter['pet_price'].history[0])

        wb.save('Results from simulations.xlsx')

        variables = [
            ('m1', 'production_volume'),
            ('m1', 'proportion_bio'),
            ('m1', 'liquidity'),
            ('m1', 'profitability'),
            ('E', 'levy_rate'),
            ('EA', 'emissions'),
            ('E', 'bio_feedstock_price'),
        ]
        # var2 : ('m1', 'bio_production'),
        # var3 : ('m1', 'fossil_production')

        for variable in variables:
            if variable[0] == 'm1':
                history = manufacturer1.parameter[variable[1]].history
            elif variable[0] == 'E':
                history = environment.parameter[variable[1]].history
            elif variable[0] == 'EA':
                history = environment.aggregate[variable[1]].history
            else:
                print("invalid variable name,", variable[0])

    return


def graph(parameter):
    assert isinstance(parameter, Parameter) or isinstance(parameter, Environment_Variable)
    y = parameter.history
    t = np.arange(0, len(y), 1)
    fig, ax1 = plt.subplots()
    ax1.plot(t, y)

    ax1.set_xlabel('Month')
    # ax1.set_ylabel('Price of bio feedstock')
    # ax1.set_ylabel('Proportion bio-PET')
    # ax1.set_ylabel('Levy rate')
    ax1.set_ylabel('Emissions')

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


def cell_write(sheet, coordinates, val):
    assert isinstance(sheet, openpyxl.worksheet.worksheet.Worksheet)
    assert type(coordinates) == tuple and len(coordinates) == 2, ('wrong tuple,', coordinates)
    assert type(coordinates[0]) == int and coordinates[0] > 0
    assert type(coordinates[1]) == int and 27 > coordinates[1] > 0

    row = coordinates[0]
    col = coordinates[1]

    col_letter = chr(col + 64)
    excel_coord = col_letter + str(row)

    sheet[excel_coord].value = val

    return
