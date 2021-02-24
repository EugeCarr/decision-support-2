"""This file defines simulation parameters for the first build of the model"""
import agent
import numpy as np


def simulate(months):
    # Initialise the manufacturer agent with all values for t = 0
    pet_manufacturer = agent.PET_Manufacturer('PET Manufacturer')

    # starting values
    pet_manufacturer.production_volume = 0
    pet_manufacturer.refresh_independents()

    pet_manufacturer.production_volume = np.float64(1000)
    pet_manufacturer.tax_rate = 0.19
    pet_manufacturer.levy_rate = 0

    pet_manufacturer.calculate_dependents()

    # make projections from t = 0
    pet_manufacturer.project_independents()
    pet_manufacturer.project_dependents()

    # Run simulation for defined number of months
    while pet_manufacturer.month <= months:
        pet_manufacturer.time_step()

    return
