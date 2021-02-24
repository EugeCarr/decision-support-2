"""This file defines the Agent class, and subclasses thereof, for the agent-based simulation"""
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

        self.month = int(0)  # current month which will be incremented at each time step

        # define current values
        self.production_volume = float  # total PET production per annum
        self.unit_sale_price = float  # sale price of one unit of PET
        self.unit_feedstock_cost = float  # feedstock cost per unit of PET produced
        self.unit_process_cost = float  # cost of running process per unit of PET produced

        # define projections
        self.production_projection = np.empty(120)  # 10 years of production volumes (floats)
        self.unit_sale_price_projection = np.empty(120)  # 10 years of sale prices (floats)
        self.unit_feedstock_cost_projection = np.empty(120)  # 10 years of feedstock costs (floats)
        self.unit_process_cost_projection = np.empty(120)  # 10 years of process costs (floats)

        # define arrays to store records
        self.production_history = np.empty(120)
        self.sale_price_history = np.empty(120)
        self.feedstock_cost_history = np.empty(120)
        self.process_cost_history = np.empty(120)
        return

# region -- methods to calculate values at the current time for each variable
    def refresh_production_volume(self):
        # production volume is defined by growth rates in 2 periods

        sim_period_0 = 5  # end year for first simulation period
        sim_period_0_months = sim_period_0 * 12  # end month for first simulation period
        growth_rate_0 = 1.02  # YoY growth rate for the first simulation period, expressed as a ratio
        growth_rate_0_monthly = np.power(growth_rate_0, 1 / 12)  # annual growth rate changed to month-on-month

        sim_period_1 = 10  # end year for second simulation period
        sim_period_1_months = sim_period_1 * 12  # end month for second simulation period
        growth_rate_1 = 1.03  # YoY growth rate for the second simulation period, expressed as a ratio
        growth_rate_1_monthly = np.power(growth_rate_1, 1 / 12)  # annual growth rate changed to month-on-month

        if self.month <= sim_period_0_months:
            self.production_volume = self.production_volume * growth_rate_0_monthly

        elif self.month <= sim_period_1_months:
            self.production_volume = self.production_volume * growth_rate_1_monthly

        return

    def refresh_unit_sale_price(self):
        # unit sale price is given by a normal distribution
        mean = float(4)
        std_dev = 0.2
        self.unit_sale_price = np.random.normal(mean, std_dev, None)
        return

    def refresh_unit_feedstock_cost(self):
        # unit feedstock cost is given by a normal distribution
        mean = float(2)
        std_dev = 0.1
        self.unit_feedstock_cost = np.random.normal(mean, std_dev, None)
        return

    def refresh_unit_process_cost(self):
        # process cost is given by a normal distribution
        mean = float(1)
        std_dev = 0.2
        self.unit_process_cost = np.random.normal(mean, std_dev, None)
        return

    def refresh_state(self):
        # calculate new values for all variables
        self.refresh_production_volume()
        self.refresh_unit_sale_price()
        self.refresh_unit_feedstock_cost()
        self.refresh_unit_process_cost()
        return

# endregion

    def record_timestep(self):
        # method to write current state variables to a record
        self.production_history[self.month] = self.production_volume
        self.sale_price_history[self.month] = self.unit_sale_price
        self.feedstock_cost_history[self.month] = self.unit_feedstock_cost
        self.process_cost_history[self.month] = self.unit_process_cost
        return

    def advance_month(self):
        self.record_timestep()
        self.refresh_state()
        return

# region -- methods to make projections into the future
    def project_volume(self):
        """This will calculate the projected PET production volume for the next 10 years,
        recording it to self.production_projection"""
        return

    def project_sale_price(self):
        # Calculate the projected PET sale price for the next 10 years
        self.unit_sale_price_projection.fill(4)  # fixed value (mean of normal dist from self.refresh_unit_sale_price)
        return

    def project_feedstock_cost(self):
        # Calculate the projected PET sale price for the next 10 years
        self.unit_feedstock_cost_projection.fill(2)  # fixed value (mean of normal dist from self.refresh_...)
        return

    def project_process_cost(self):
        # Calculate the projected PET sale price for the next 10 years
        self.unit_process_cost_projection.fill(1)  # fixed value (mean of normal dist from self.refresh_...)
        return

    def make_projections(self):
        self.project_volume()
        self.project_sale_price()
        self.project_feedstock_cost()
        self.project_process_cost()
        return
# endregion
