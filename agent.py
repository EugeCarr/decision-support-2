"""This file defines the Agent class, and subclasses thereof, for the agent-based simulation"""
import numpy as np


class Agent(object):
    def __init__(self, name, sim_time):
        assert type(name) == str, ('name must be a string. input value is a', type(name))
        assert type(sim_time) == int, ('sim_time must be an integer. input value is a', type(sim_time))
        assert sim_time > 0, 'sim_time must be greater than zero'
        self.name = name
        return


class PET_Manufacturer(Agent):
    # object initialisation
    def __init__(self, name, sim_time):
        super().__init__(name, sim_time)

        self.month = int(0)  # current month which will be incremented at each time step

        # define independent variables for current time
        self.production_volume = np.float64()  # total PET production per annum
        self.unit_sale_price = np.float64()  # sale price of one unit of PET
        self.unit_feedstock_cost = np.float64()  # feedstock cost per unit of PET produced
        self.unit_process_cost = np.float64  # cost of running process per unit of PET produced

        self.tax_rate = np.float64  # current tax on profits
        self.levy_rate = np.float64  # current levy on production/emission/consumption/etc.

        # define dependent variables
        self.gross_profit = np.float64  # profits prior to taxes and levies
        self.tax_payable = np.float64
        self.levies_payable = np.float64
        self.net_profit = np.float64  # monthly profit after tax and levies
        self.projection_met = 0

        self.projection_time = 120  # how many months into the future will be predicted?

        # define projections
        self.production_projection = np.zeros(self.projection_time)  # 10 years of production volumes (floats)
        self.unit_sale_price_projection = np.zeros(self.projection_time)  # 10 years of sale prices (floats)
        self.unit_feedstock_cost_projection = np.zeros(self.projection_time)  # 10 years of feedstock costs (floats)
        self.unit_process_cost_projection = np.zeros(self.projection_time)  # 10 years of process costs (floats)

        self.gross_profit_projection = np.zeros(self.projection_time)
        self.tax_payable_projection = np.zeros(self.projection_time)
        self.levies_payable_projection = np.zeros(self.projection_time)
        self.net_profit_projection = np.zeros(self.projection_time)

        self.tax_rate_projection = np.zeros(self.projection_time)  # tax rate projection for 10 years (floats)
        self.levy_projection = np.zeros(self.projection_time)  # levy rate projection for 10 years (floats)

        # define arrays to store records
        history_length = sim_time
        self.production_history = np.zeros(history_length)
        self.sale_price_history = np.zeros(history_length)
        self.feedstock_cost_history = np.zeros(history_length)
        self.process_cost_history = np.zeros(history_length)
        self.gross_profit_history = np.zeros(history_length)
        self.tax_history = np.zeros(history_length)
        self.levy_history = np.zeros(history_length)
        self.net_profit_history = np.zeros(history_length)
        self.projection_met_history = np.zeros(history_length)

        return

    # region -- methods to calculate values at the current time for each independent variable
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

        if self.month == 0:
            self.production_volume = 1000  # production volume starts at 1000 units per annum

        elif self.month <= sim_period_0_months:
            self.production_volume = self.production_volume * growth_rate_0_monthly

        elif self.month <= sim_period_1_months:
            self.production_volume = self.production_volume * growth_rate_1_monthly

        else:
            raise ValueError('production growth not defined for month', self.month)

        return

    def refresh_unit_sale_price(self):
        # unit sale price is given by a normal distribution
        mean = float(4)
        std_dev = 0.01
        self.unit_sale_price = np.random.normal(mean, std_dev, None)
        return

    def refresh_unit_feedstock_cost(self):
        # unit feedstock cost is given by a normal distribution
        mean = float(2)
        std_dev = 0.01
        self.unit_feedstock_cost = np.random.normal(mean, std_dev, None)
        return

    def refresh_unit_process_cost(self):
        # process cost is given by a normal distribution around a mean which is a weakly decreasing function of
        # production volume, such that a doubling in production reduces processing unit cost by 10%, starting from 1
        mean = 2.8576 / np.power(self.production_volume, 0.152)
        std_dev = 0.005
        self.unit_process_cost = np.random.normal(mean, std_dev, None)
        return

    def refresh_independents(self):
        # calculate new values for all variables
        self.refresh_production_volume()
        self.refresh_unit_sale_price()
        self.refresh_unit_feedstock_cost()
        self.refresh_unit_process_cost()
        return

    # endregion

    # region -- methods for dependent variables
    def calculate_gross_profit(self):
        production_in_month = self.production_volume / 12
        revenue = production_in_month * self.unit_sale_price
        costs = production_in_month * (self.unit_feedstock_cost + self.unit_process_cost)
        self.gross_profit = revenue - costs
        return

    def calculate_tax_payable(self):
        self.tax_payable = self.gross_profit * self.tax_rate
        return

    def calculate_levies_payable(self):
        """This will calculate the levies payable on production/consumption/emission, once they are defined"""
        self.levies_payable = 0
        return

    def calculate_net_profit(self):
        self.net_profit = self.gross_profit - (self.tax_payable + self.levies_payable)
        return

    def calculate_dependents(self):
        self.calculate_gross_profit()
        self.calculate_tax_payable()
        self.calculate_levies_payable()
        self.calculate_net_profit()
        return

    # endregion

    def record_timestep(self):
        # method to write current variables (independent and dependent) to records
        self.production_history[self.month] = self.production_volume
        self.sale_price_history[self.month] = self.unit_sale_price
        self.feedstock_cost_history[self.month] = self.unit_feedstock_cost
        self.process_cost_history[self.month] = self.unit_process_cost
        self.gross_profit_history[self.month] = self.gross_profit
        self.tax_history[self.month] = self.tax_payable
        self.levy_history[self.month] = self.levies_payable
        self.net_profit_history[self.month] = self.net_profit
        self.projection_met_history[self.month] = self.projection_met
        return

    def update_current_state(self):
        # methods to be called every time the month is advanced
        self.refresh_independents()
        self.calculate_dependents()
        return

    # region -- methods for making projections into the future
    def project_volume(self):
        """This will calculate the projected (annualised) PET production volume for each month for 10 years,
        recording it to self.production_projection"""
        predicted_annual_growth_rate = 1.02
        monthly_growth_rate = np.power(predicted_annual_growth_rate, 1/12)
        initial_volume = self.production_volume
        for i in range(self.projection_time):
            self.production_projection[i] = initial_volume * pow(monthly_growth_rate, i)

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

    def project_gross_profit(self):
        # calculate revenues and costs at each month
        monthly_production_projection = self.production_projection / 12

        revenue_projection = np.multiply(monthly_production_projection, self.unit_sale_price_projection)
        feed_cost_projection = np.multiply(monthly_production_projection, self.unit_feedstock_cost_projection)
        process_cost_projection = np.multiply(monthly_production_projection, self.unit_process_cost_projection)
        total_cost_projection = np.add(feed_cost_projection, process_cost_projection)

        self.gross_profit_projection = np.subtract(revenue_projection, total_cost_projection)
        return

    def project_tax_payable(self):
        self.tax_payable_projection = np.multiply(self.gross_profit_projection, self.tax_rate_projection)
        return

    def project_levies_payable(self):
        """This will calculate projected levies payable, once they are defined."""
        length = len(self.levies_payable_projection)
        self.levies_payable_projection = np.zeros(length)
        return

    def project_net_profit(self):
        p_0 = self.gross_profit_projection
        p_1 = np.subtract(p_0, self.tax_payable_projection)
        p_2 = np.subtract(p_1, self.levies_payable_projection)
        self.net_profit_projection = p_2
        return

    def project_independents(self):
        # calculate projections for independent variables
        self.project_volume()
        self.project_sale_price()
        self.project_feedstock_cost()
        self.project_process_cost()
        return

    def project_dependents(self):
        # calculate projections for dependent variables (i.e. must run after self.project_independents)
        self.project_gross_profit()
        self.project_tax_payable()
        self.project_levies_payable()
        self.project_net_profit()
        return

    def new_projection(self):
        self.project_independents()
        self.project_dependents()
        return
    # endregion

    def projection_check(self):
        # checks whether the next target will be met on the basis of latest projection
        # monthly profit targets at years 5 and 10
        target_year_5 = 80
        target_year_10 = 85

        time_to_yr5 = 60 - self.month
        time_to_yr_10 = 120 - self.month

        if time_to_yr5 > 0:
            if self.net_profit_projection[time_to_yr5] >= target_year_5:
                self.projection_met = 1
            else:
                self.projection_met = 0

        elif time_to_yr_10 > 0:
            if self.net_profit_projection[time_to_yr_10] >= target_year_10:
                self.projection_met = 1
            else:
                self.projection_met = 0

        else:
            pass

        return

    def time_step(self):
        self.month += 1
        self.update_current_state()
        if self.month % 12 == 0:
            self.new_projection()
            self.projection_check()
        self.record_timestep()
        return
