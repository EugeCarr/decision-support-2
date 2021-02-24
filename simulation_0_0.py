"""This file defines simulation parameters for the first build of the model"""
import agent
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt


def simulate(months, table=bool, plot=bool):
    # Initialise the manufacturer agent with all values for t = 0
    pet_manufacturer = agent.PET_Manufacturer('PET Manufacturer', months)

    # starting values
    pet_manufacturer.production_volume = 0
    pet_manufacturer.refresh_independents()

    pet_manufacturer.production_volume = np.float64(1000)
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
    if table:
        t = np.arange(0, months, 1)

        table = []
        for i in range(0, months):
            table.append([t[i], pet_manufacturer.projection_met_history[i]])

        headers = ["Month", "Profit"]
        print(tabulate(table, headers))

    if plot:
        y = pet_manufacturer.projection_met_history
        x = t
        plt.plot(x, y)
        plt.xlabel('Month')
        plt.ylabel('Profit')
        plt.show()

    return
