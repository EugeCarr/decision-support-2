"""This file defines simulation parameters for the first build of the model"""
import agent
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt


def simulate(months, table=bool, plot=bool):
    # Initialise the manufacturer agent with all values for t = 0
    pet_manufacturer = agent.PET_Manufacturer('PET Manufacturer', months)

    # starting values
    pet_manufacturer.production_volume = np.float64(1000)
    pet_manufacturer.refresh_independents()

    pet_manufacturer.tax_rate = 0.19
    pet_manufacturer.levy_rate = 0

    pet_manufacturer.calculate_dependents()

    # make projections from t = 0
    pet_manufacturer.new_projection()
    pet_manufacturer.projection_check()

    pet_manufacturer.record_timestep()

    # Run simulation for defined number of months
    while pet_manufacturer.month < months - 1:
        pet_manufacturer.time_step()

    # data output & analysis
    t = np.arange(0, months, 1)
    if table:
        table = []
        for i in range(0, months):
            table.append([t[i], pet_manufacturer.net_profit_history[i]])

        headers = ["Month", "Profit"]
        print(tabulate(table, headers))

    if plot:
        y = pet_manufacturer.process_cost_history
        y1 = pet_manufacturer.production_history
        x = t
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)
        ax2 = ax1.twinx()
        ax2.plot(x, y1, color='red')

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Processing unit cost')
        ax2.set_ylabel('Production Volume')

        fig.tight_layout()
        plt.show()

    return
