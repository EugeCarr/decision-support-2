"""This file defines simulation parameters for the first build of the model"""
import agent
from regulator import Regulator
from regulator import Policy
import numpy as np
import copy
from tabulate import tabulate
from matplotlib import pyplot as plt


def simulate(months, table=bool, plot=bool):
    # create agents and specify their parameters
    pet_manufacturer = agent.PET_Manufacturer('PET Manufacturer', months)

    policy = Policy()
    policy.add_level([])

    regulator = Regulator('Regulator', months, int(12), policy)

    # Run simulation for defined number of months
    while pet_manufacturer.month < months:
        pet_manufacturer.time_step()

    # data output & analysis
    t = np.arange(0, months, 1)

    if table:
        table = []
        for i in range(0, months):
            table.append([t[i],
                          pet_manufacturer.projection_met_history[i],
                          pet_manufacturer.profitability_history[i],
                          pet_manufacturer.bio_history[i]])

        headers = ["Month", "Projection met?", "Profitability", "Bio Proportion"]
        print(tabulate(table, headers))

    if plot:
        y = pet_manufacturer.bio_history
        x = t
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('bio proportion')

        fig.tight_layout()
        plt.show()

    return
