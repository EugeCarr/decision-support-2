"""This file defines the Agent class, and subclasses thereof, for the agent-based simulation"""
import numpy as np


def run_check():
    global proceed
    while True:
        try:
            proceed = str(input('Do you want to proceed? Y/N:'))
        except ValueError:
            continue

        if proceed.lower() not in 'yn':
            continue
        else:
            break

    if proceed.lower() == 'n':
        raise SystemExit(0)
    elif proceed.lower() == 'y':
        pass


class Agent(object):
    def __init__(self, name, sim_time):
        assert type(name) == str, ('name must be a string. input value is a', type(name))
        assert type(sim_time) == int, ('sim_time must be an integer. input value is a', type(sim_time))
        assert sim_time > 0, 'sim_time must be greater than zero'
        self.name = name

        print(' Object', self.name, 'created \n ================================')
        return


class PET_Manufacturer(Agent):
    # object initialisation
    def __init__(self, name, sim_time):
        super().__init__(name, sim_time)

        self.month = int(0)  # current month which will be incremented at each time step

        # define independent variables for current time
        self.production_volume = np.float64(1000)  # total PET production per annum, starts at 1000
        self.unit_sale_price = np.float64()  # sale price of one unit of PET
        self.unit_feedstock_cost = np.float64()  # feedstock cost per unit of PET produced
        self.unit_process_cost = np.float64()  # cost of running process per unit of PET produced

        self.proportion_bio = np.float64(0)  # proportion of production from biological feedstocks
        self.bio_feedstock_cost = np.float64()  # bio feedstock cost per unit of PET produced
        self.bio_process_cost = np.float64()  # cost of process per unit of PET from bio routes, starts at 1.5

        self.tax_rate = np.float64(0.19)  # current tax on profits, starts at 19%
        self.levy_rate = np.float64(0.2)  # current levy on production/emission/consumption/etc., starts at zero
        self.emissions_rate = np.float64(5)  # units of emissions per unit of PET produced from non-bio route

        # define dependent variables
        self.gross_profit = np.float64()  # profits prior to taxes and levies
        self.tax_payable = np.float64()
        self.levies_payable = np.float64()
        self.net_profit = np.float64()  # monthly profit after tax and levies
        self.projection_met = 1  # 1 or 0 depending on whether the next target will be met by current projection
        self.emissions = np.float64()  # emissions from manufacturing PET from fossil fuels
        self.profitability = np.float64()  # profitability (net profit per unit production)

        self.projection_time = 120  # how many months into the future will be predicted?

        # define projections
        self.production_projection = np.zeros(self.projection_time)  # 10 years of production volumes (floats)
        self.unit_sale_price_projection = np.zeros(self.projection_time)  # 10 years of sale prices (floats)
        self.unit_feedstock_cost_projection = np.zeros(self.projection_time)  # 10 years of feedstock costs (floats)
        self.unit_process_cost_projection = np.zeros(self.projection_time)  # 10 years of process costs (floats)
        self.proportion_bio_projection = np.zeros(self.projection_time)
        self.bio_feedstock_cost_projection = np.zeros(self.projection_time)
        self.bio_process_cost_projection = np.zeros(self.projection_time)

        self.gross_profit_projection = np.zeros(self.projection_time)
        self.emissions_projection = np.zeros(self.projection_time)
        self.tax_payable_projection = np.zeros(self.projection_time)
        self.levies_payable_projection = np.zeros(self.projection_time)
        self.net_profit_projection = np.zeros(self.projection_time)
        self.profitability_projection = np.zeros(self.projection_time)

        self.tax_rate_projection = np.ones(self.projection_time) * self.tax_rate
        self.levy_projection = np.ones(self.projection_time) * self.levy_rate

        # define arrays to store records
        history_length = sim_time
        self.production_history = np.zeros(history_length)
        self.sale_price_history = np.zeros(history_length)
        self.feedstock_cost_history = np.zeros(history_length)
        self.process_cost_history = np.zeros(history_length)
        self.bio_history = np.zeros(history_length)
        self.bio_feedstock_history = np.zeros(history_length)
        self.bio_process_history = np.zeros(history_length)
        self.gross_profit_history = np.zeros(history_length)
        self.emissions_history = np.zeros(history_length)
        self.tax_history = np.zeros(history_length)
        self.levy_history = np.zeros(history_length)
        self.net_profit_history = np.zeros(history_length)
        self.projection_met_history = np.zeros(history_length)
        self.bio_target_history = np.zeros(history_length)
        self.profitability_history = np.zeros(history_length)

        # define variables for the targets against which projections are measured
        # and the times at which they happen
        self.target1_value = np.float64(1.5)  # currently fixed values
        self.target1_year = 5

        self.target2_value = np.float64(1.6)  # currently fixed values
        self.target2_year = 10

        self.beyond_target_range = False  # a boolean set to true if the simulation runs beyond the point for which
        # targets are defined

        self.invest_in_bio = False  # set to True if investment in bio route starts
        self.proportion_bio_target = np.float64(0)  # target value for the proportion of production via bio route
        self.proportion_change_rate = np.float64(0.1 / 15)  # greatest possible monthly change in self.proportion_bio
        self.implementation_delay = int(3)  # time delay between investment decision and movement of bio_proportion
        self.implementation_countdown = int(0)  # countdown to change of direction
        self.under_construction = False  # is change in bio capacity occurring?

        # output initialisation state to console
        print(' INITIAL STATE \n -------------'
              '\n Annual production volume:', self.production_volume,
              '\n Corporation tax rate:', self.tax_rate,
              '\n Levy rate:', self.levy_rate,
              '\n Projection horizon (months):', self.projection_time,
              '\n Target at year', self.target1_year, ':', self.target1_value,
              '\n Target at year', self.target2_year, ':', self.target2_value,
              '\n -------------')

        # run_check()

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

        if self.month <= sim_period_0_months:
            self.production_volume = self.production_volume * growth_rate_0_monthly

        elif self.month <= sim_period_1_months:
            self.production_volume = self.production_volume * growth_rate_1_monthly

        else:
            raise ValueError('production growth not defined for month', self.month)

        return

    def refresh_unit_sale_price(self):
        # unit sale price is given by a normal distribution
        mean = float(6)
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

    def refresh_proportion_bio(self):
        # monthly change in bio proportion is either the amount to reach the target value, or else the maximum change
        if self.invest_in_bio:
            if self.implementation_countdown == 0 and self.proportion_bio != self.proportion_bio_target:
                self.under_construction = True
                distance_from_target = self.proportion_bio_target - self.proportion_bio
                if abs(distance_from_target) < self.proportion_change_rate:
                    self.proportion_bio = self.proportion_bio_target
                elif distance_from_target > 0:
                    self.proportion_bio += self.proportion_change_rate
                elif distance_from_target < 0:
                    self.proportion_bio -= self.proportion_change_rate
                else:
                    pass
            else:
                pass
        else:
            self.under_construction = False
        return

    def refresh_bio_feedstock_cost(self):
        # unit feedstock cost is given by a normal distribution
        mean = float(2)
        std_dev = 0.01
        self.bio_feedstock_cost = np.random.normal(mean, std_dev, None)
        return

    def refresh_bio_process_cost(self):
        # process cost is given by a normal distribution
        mean = 1.05
        std_dev = 0.005
        self.bio_process_cost = np.random.normal(mean, std_dev, None)
        return

    def refresh_independents(self):
        # calculate new values for all variables
        self.refresh_production_volume()
        self.refresh_unit_sale_price()
        self.refresh_unit_feedstock_cost()
        self.refresh_unit_process_cost()
        self.refresh_proportion_bio()
        self.refresh_bio_feedstock_cost()
        self.refresh_bio_process_cost()
        return

    # endregion

    # region -- methods for dependent variables
    def calculate_gross_profit(self):
        production_in_month = self.production_volume / 12
        revenue = production_in_month * self.unit_sale_price
        costs = production_in_month * ((1 - self.proportion_bio) * (self.unit_feedstock_cost + self.unit_process_cost) +
                                       self.proportion_bio * (self.bio_feedstock_cost + self.bio_process_cost))
        self.gross_profit = revenue - costs
        return

    def calculate_emissions(self):
        fossil_production = self.production_volume / 12 * (1 - self.proportion_bio)
        self.emissions = fossil_production * self.emissions_rate
        return

    def calculate_tax_payable(self):
        self.tax_payable = self.gross_profit * self.tax_rate
        return

    def calculate_levies_payable(self):
        """This will calculate the levies payable on production/consumption/emission, once they are defined"""
        self.levies_payable = self.levy_rate * self.emissions
        return

    def calculate_net_profit(self):
        self.net_profit = self.gross_profit - (self.tax_payable + self.levies_payable)
        return

    def calculate_profitability(self):
        self.profitability = self.net_profit / (self.production_volume / 12)

    def calculate_dependents(self):
        self.calculate_gross_profit()
        self.calculate_emissions()
        self.calculate_tax_payable()
        self.calculate_levies_payable()
        self.calculate_net_profit()
        self.calculate_profitability()
        return

    # endregion

    def record_timestep(self):
        # method to write current variables (independent and dependent) to records
        self.production_history[self.month] = self.production_volume
        self.sale_price_history[self.month] = self.unit_sale_price
        self.feedstock_cost_history[self.month] = self.unit_feedstock_cost
        self.process_cost_history[self.month] = self.unit_process_cost
        self.bio_history[self.month] = self.proportion_bio
        self.bio_feedstock_history[self.month] = self.bio_feedstock_cost
        self.bio_process_history[self.month] = self.bio_process_cost
        self.gross_profit_history[self.month] = self.gross_profit
        self.emissions_history[self.month] = self.emissions
        self.tax_history[self.month] = self.tax_payable
        self.levy_history[self.month] = self.levies_payable
        self.net_profit_history[self.month] = self.net_profit
        self.projection_met_history[self.month] = self.projection_met
        self.bio_target_history[self.month] = self.proportion_bio_target
        self.profitability_history[self.month] = self.profitability
        return

    def update_current_state(self):
        # methods to be called every time the month is advanced
        self.refresh_independents()
        self.calculate_dependents()
        return

    # region -- methods for making projections into the future
    def project_volume(self):
        # calculates the projected (annualised) PET production volume for each month,
        # recording it to self.production_projection
        predicted_annual_growth_rate = 1.02
        monthly_growth_rate = np.power(predicted_annual_growth_rate, 1 / 12)
        initial_volume = self.production_volume
        # calculated using a fixed month-on-month growth rate from the most recent production volume
        for i in range(self.projection_time):
            self.production_projection[i] = initial_volume * pow(monthly_growth_rate, i)

        return

    def project_sale_price(self):
        # Calculate the projected PET sale prices
        self.unit_sale_price_projection.fill(6)  # fixed value (mean of normal dist from self.refresh_unit_sale_price)
        return

    def project_feedstock_cost(self):
        # Calculate the projected PET feedstock costs
        self.unit_feedstock_cost_projection.fill(2)  # fixed value (mean of normal dist from self.refresh_...)
        return

    def project_process_cost(self):
        # Calculate the projected PET processing costs
        self.unit_process_cost_projection.fill(1)  # fixed value (mean of normal dist from self.refresh_...)
        return

    def project_proportion_bio(self):
        # projection of proportion of production from bio routes
        time_to_target = int(np.ceil((self.proportion_bio_target - self.proportion_bio) / self.proportion_change_rate)
                             + self.implementation_countdown)
        self.proportion_bio_projection.fill(self.proportion_bio_target)

        if time_to_target > 1:

            for i in range(time_to_target - 1):
                try:
                    self.proportion_bio_projection[i] = self.proportion_bio + self.proportion_change_rate * (i + 1)
                except IndexError:
                    print('time to reach target bio proportion is longer than', self.projection_time, 'months')
                    print('behaviour in these conditions is undefined. aborting simulation')
                    raise SystemExit(0)
        else:
            pass
        return

    def project_bio_feedstock_cost(self):
        self.bio_feedstock_cost_projection.fill(2)
        return

    def project_bio_process_cost(self):
        self.bio_process_cost_projection.fill(1.05)
        return

    def project_emissions(self):
        monthly_production_projection = self.production_projection / 12
        self.emissions_projection = np.multiply(
            monthly_production_projection, np.subtract(
                np.ones(self.projection_time), self.proportion_bio_projection)
        ) * self.emissions_rate
        return

    def project_gross_profit(self):
        # calculate revenues and costs at each month
        monthly_production_projection = self.production_projection / 12

        revenue_projection = np.multiply(monthly_production_projection, self.unit_sale_price_projection)

        fossil_cost_projection = np.multiply(
            np.add(np.multiply(monthly_production_projection, self.unit_feedstock_cost_projection),
                   np.multiply(monthly_production_projection, self.unit_process_cost_projection)),
            np.subtract(np.ones(self.projection_time), self.proportion_bio_projection))
        bio_cost_projection = np.multiply(
            np.add(np.multiply(monthly_production_projection, self.bio_feedstock_cost_projection),
                   np.multiply(monthly_production_projection, self.bio_process_cost_projection)),
            self.proportion_bio_projection)

        total_cost_projection = np.add(fossil_cost_projection, bio_cost_projection)

        self.gross_profit_projection = np.subtract(revenue_projection, total_cost_projection)
        return

    def project_tax_payable(self):
        self.tax_payable_projection = np.multiply(self.gross_profit_projection, self.tax_rate_projection)
        return

    def project_levies_payable(self):
        self.levies_payable_projection = np.multiply(self.emissions_projection, self.levy_projection)
        return

    def project_net_profit(self):
        p_0 = self.gross_profit_projection
        p_1 = np.subtract(p_0, self.tax_payable_projection)
        p_2 = np.subtract(p_1, self.levies_payable_projection)
        self.net_profit_projection = p_2
        return

    def project_profitability(self):
        self.profitability_projection = np.divide(self.net_profit_projection, self.production_projection / 12)

    def project_independents(self):
        # calculate projections for independent variables
        self.project_volume()
        self.project_sale_price()
        self.project_feedstock_cost()
        self.project_process_cost()
        self.project_proportion_bio()
        self.project_bio_feedstock_cost()
        self.project_bio_process_cost()
        return

    def project_dependents(self):
        # calculate projections for dependent variables (i.e. must run after self.project_independents)
        # order of operations for these methods may be important - CHECK
        self.project_gross_profit()
        self.project_emissions()
        self.project_tax_payable()
        self.project_levies_payable()
        self.project_net_profit()
        self.project_profitability()
        return

    def new_projection(self):
        self.project_independents()
        self.project_dependents()
        return

    # endregion

    def projection_check(self):
        # checks whether the next target will be met on the basis of latest projection
        # monthly profit targets at years 5 and 10

        time_to_yr1 = self.target1_year * 12 - self.month
        time_to_yr2 = self.target2_year * 12 - self.month

        if time_to_yr1 > 0:
            if self.profitability_projection[time_to_yr1] >= self.target1_value:
                self.projection_met = 1
            else:
                self.projection_met = 0

        elif time_to_yr2 > 0:
            if self.profitability_projection[time_to_yr2] >= self.target2_value:
                self.projection_met = 1
            else:
                self.projection_met = 0

        else:
            # if there is no target defined into the future, PROJECTION_MET is set to 0
            self.projection_met = 0

            if not self.beyond_target_range:
                print('No target defined beyond month', self.month)
                self.beyond_target_range = True

        return

    def investment_decision(self):
        # decision logic for increasing investment in biological process route
        if self.projection_met == 1:
            pass
        elif self.projection_met == 0:
            if not self.invest_in_bio:
                self.invest_in_bio = True

            while self.proportion_bio_target <= 0.9 and self.projection_met == 0:
                self.implementation_countdown = self.implementation_delay
                self.proportion_bio_target += 0.1
                self.new_projection()
                self.projection_check()
                if self.proportion_bio_target == 1 and self.projection_met == 0:
                    print('Month:', self.month, '\n next profitability target could not be met'
                                                'at any bio proportion target')

            else:
                pass

        else:
            pass

        return

    def time_step(self):
        if self.implementation_countdown > 0:
            self.implementation_countdown -= 1

        self.update_current_state()
        if self.month % 12 == 0:
            self.new_projection()
            self.projection_check()
            self.investment_decision()
        self.record_timestep()
        self.month += 1
        return
