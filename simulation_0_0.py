"""This file defines simulation parameters for the first build of the model"""
import agent
from regulator import Regulator
from regulator import Policy
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt


def simulate(months, table=bool, plot=bool):
    # create agents and specify their parameters
    month = int(0)

    pet_manufacturer = agent.PET_Manufacturer('PET Manufacturer', months)

    policy = Policy()
    policy.add_level([1900, 0.19, 0.2])
    policy.add_level([2000, 0.19, 0.225])
    policy.add_level([2050, 0.19, 0.25])

    notice_period = int(12)

    regulator = Regulator('Regulator', months, notice_period, policy)

    # Run simulation for defined number of months
    while month < months:
        # advance time counter in each agent
        pet_manufacturer.month = month
        regulator.month = month

        # execute standard monthly routines
        pet_manufacturer.time_step()
        regulator.iterate_regulator(pet_manufacturer.emissions.value)

        # if the regulator rate has just changed (resulting in mismatch between agents) then update it
        if pet_manufacturer.levy_rate != regulator.levy_rate:
            pet_manufacturer.levy_rate = regulator.levy_rate
            pet_manufacturer.time_to_levy_change = regulator.time_to_change
            pet_manufacturer.levy_rate_changing = False

        # if a change in the levy rate is approaching, tell the pet_manufacturer
        if regulator.changing:
            pet_manufacturer.levy_rate_changing = True
            pet_manufacturer.time_to_levy_change = regulator.time_to_change
            pet_manufacturer.future_levy_rate = regulator.pol_table[regulator.level + 1][2]
        else:
            pass

        month += 1

    # data output & analysis
    t = np.arange(0, months, 1)

    if table:
        table = []
        for i in range(0, months):
            table.append([t[i],
                          pet_manufacturer.production_volume.history[i]])

        headers = ["Month", "Production Volume"]
        print(tabulate(table, headers))

    if plot:
        y = pet_manufacturer.emissions.history
        x = t
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Emissions')

        fig.tight_layout()
        plt.show()

    return
