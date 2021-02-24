"""This file defines the class of agent for the agent-based simulation"""
import numpy as np


class Agent(object):
    def __init__(self, name):
        assert type(name) == str, ('name must be a string. input value is a', type(name))
        self.name = name
        return


class PET_Manufacturer(Agent):
    # object initialisation
    def __init__(self, name):
        super().__init__(name)

        # define current values
        self.production_volume = float  # total PET production per annum
        self.unit_sale_price = float  # sale price of one unit of PET
        self.unit_feedstock_cost = float  # feedstock cost per unit of PET produced
        self.unit_process_cost = float  # cost of running process per unit of PET produced

        # define projections
        self.production_growth_projection = np.empty(120)  # 10 years of month-on-month growth %
        self.unit_sale_price_projection = np.empty(120)  # 10 years of sale prices (floats)
        self.unit_feedstock_cost_projection = np.empty(120)  # 10 years of feedstock costs (floats)
        self.unit_process_cost_projection = np.empty(120)  # 10 years of process costs (floats)
        return
