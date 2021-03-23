"""This file defines simulation parameters for the first build of the model"""
import agent
from regulator import Regulator
from regulator import Policy
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import copy


def simulate(months, table=bool, plot=bool):
    # create agents and specify their parameters
    month = int(0)

    # self.production_volume = Parameter(production_volume, production_volume_projection, sim_time,
    #                                    init=initial_production_volume)
    # # total PET production per annum, starts at 1000
    # self.unit_sale_price = Parameter(unit_sale_price, unit_sale_price_projection, sim_time)
    # # sale price of one unit of PET
    # self.unit_feedstock_cost = Parameter(unit_feedstock_cost, unit_feedstock_cost_projection, sim_time)
    # # feedstock cost per unit of PET produced
    # self.unit_process_cost = Parameter(unit_process_cost, unit_process_cost_projection, sim_time)
    # # cost of running process per unit of PET produced
    #
    # self.bio_feedstock_cost = Parameter(bio_feedstock_cost, bio_feedstock_cost_projection, sim_time)
    # # bio feedstock cost per unit of PET produced
    # self.bio_process_cost = Parameter(bio_process_cost, bio_process_cost_projection, sim_time)
    # # cost of process per unit of PET from bio routes
    # self.proportion_bio = Parameter(proportion_bio, proportion_bio_projection, sim_time)
    # # proportion of production from biological feedstocks
    # self.levy_rate = Parameter(levy_rate, levy_rate_projection, sim_time, init=np.float64(0.2))
    #
    # self.bio_capacity = Parameter(bio_capacity, bio_capacity_projection, sim_time)
    # self.fossil_capacity = Parameter(fossil_capacity, fossil_capacity_projection, sim_time,
    #                                  init=initial_production_volume)
    #
    # self.expansion_cost = Parameter(expansion_cost, expansion_cost_projection, sim_time)
    # # cost of increasing production capacity
    # self.gross_profit = Parameter(gross_profit, gross_profit_projection, sim_time)
    # # profits after levies and before taxes
    # self.emissions = Parameter(emissions, emissions_projection, sim_time)
    # # emissions from manufacturing PET from fossil fuels
    # self.tax_payable = Parameter(tax_payable, tax_payable_projection, sim_time)
    # self.levies_payable = Parameter(levies_payable, levies_payable_projection, sim_time)
    # self.net_profit = Parameter(net_profit, net_profit_projection, sim_time)
    # # monthly profit after tax and levies
    # self.profitability = Parameter(profitability, profitability_projection, sim_time)
    # # profitability (net profit per unit production)
    # self.liquidity = Parameter(liquidity, liquidity_projection, sim_time, init=np.float64(5000))
    # # accumulated cash
    # self.profit_margin = Parameter(profit_margin, profit_margin_projection, sim_time)
    #
    # # dictionary of all variables in the order in which they should be computed
    # self.parameter = {
    #     'production_volume': self.production_volume,
    #     'unit_sale_price': self.unit_sale_price,
    #     'unit_feedstock_cost': self.unit_feedstock_cost,
    #     'unit_process_cost': self.unit_process_cost,
    #     'bio_feedstock_cost': self.bio_feedstock_cost,
    #     'bio_process_cost': self.bio_process_cost,
    #     'proportion_bio': self.proportion_bio,
    #     'levy_rate': self.levy_rate,
    #     'bio_capacity': self.bio_capacity,
    #     'fossil_capacity': self.fossil_capacity,
    #     'expansion_cost': self.expansion_cost,
    #     'emissions': self.emissions,
    #     'levies_payable': self.levies_payable,
    #     'gross_profit': self.gross_profit,
    #     'tax_payable': self.tax_payable,
    #     'net_profit': self.net_profit,
    #     'profitability': self.profitability,
    #     'liquidity': self.liquidity,
    #     'profit_margin': self.profit_margin
    # }

    pet_manufacturer = agent.Manufacturer('PET Manufacturer', months)

    policy = Policy()
    policy.add_level([1900, 0.19, 0.2])
    policy.add_level([2000, 0.19, 0.225])
    policy.add_level([2100, 0.19, 0.25])
    policy.add_level([2200, 0.19, 0.275])

    notice_period = int(18)

    regulator = Regulator('Regulator', months, notice_period, policy)

    agents = [
        pet_manufacturer,
        regulator
    ]

    # Run simulation for defined number of months
    while month < months:
        # advance time counter in each agent
        for entity in agents:
            entity.month = month

        # execute standard monthly routines
        pet_manufacturer.time_step()
        regulator.iterate_regulator(pet_manufacturer.emissions.value)

        # if the regulator rate has just changed (resulting in mismatch between agents) then update it
        if pet_manufacturer.parameter['levy_rate'].value != regulator.levy_rate:
            pet_manufacturer.parameter['levy_rate'].value = regulator.levy_rate
            pet_manufacturer.time_to_levy_change = copy.deepcopy(regulator.time_to_change)
            pet_manufacturer.levy_rate_changing = False

        # if a change in the levy rate is approaching, tell the pet_manufacturer
        if regulator.changing:
            pet_manufacturer.levy_rate_changing = True
            pet_manufacturer.time_to_levy_change = copy.deepcopy(regulator.time_to_change)
            pet_manufacturer.future_levy_rate = regulator.pol_table[regulator.level + 1][2]
        else:
            pass

        month += 1

    print(' ============ \n FINAL STATE \n ============',
          '\n Regulation level:', regulator.level,
          '\n Levy rate:', regulator.levy_rate,
          '\n Bio proportion', pet_manufacturer.proportion_bio.value)

    # data output & analysis
    t = np.arange(0, months, 1)

    if table:
        table = []
        for i in range(0, months):
            table.append([t[i],
                          pet_manufacturer.profitability.history[i]])

        headers = ['Month', 'Profitability']
        print(tabulate(table, headers))

    if plot:
        y = pet_manufacturer.parameter['proportion_bio'].history
        x = t
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('')

        fig.tight_layout()
        plt.show()

    return
