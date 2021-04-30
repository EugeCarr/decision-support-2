from agent import Agent
import numpy as np
# import math


class Supplier(Agent):
    """So the feedstock supplier will continually create resource X, and supply whatever of Resource X
    there is to the Manufacturers, and set set prices for Resource X based on the availability of said resource. The
    idea is that the resource will be a renewable one, for example trees. This means that changes to amount in trees,
    will take 10 years to be realised.

    Also needs to communicate to PET manufacturers that there is no more growth in reserves once the max amount has been
    reached. After this point the companies will have to consider resource scarcity increasing the price of feedstock.

    Also needs to communicate that the price is changing when that happens"""

    def __init__(self, name, sim_time, env, start_price, elasticity=0.1):
        super().__init__(name, sim_time, env)
        assert type(start_price) == float, ("start_price should be a float, not a", type(start_price))
        assert type(elasticity) == float, ("start_price should be a float, not a", type(elasticity))

        self.start_price = start_price

        self.price = start_price

        self.price_elasticity = elasticity

        self.demand = np.float()
        self.demand_baseline = np.float()
        self.demand_history = []
        self.ratio_baseline = np.float()

        self.random_switch = True

        return

    def get_demand(self):

        self.demand = self.env.aggregate['bio_feedstock_consumption'].value
        if len(self.demand_history) == 0:
            self.demand_baseline = np.float(self.env.aggregate['bio_feedstock_consumption'].value)

        self.demand_history.append(self.env.aggregate['bio_feedstock_consumption'].value)
        return

    def set_price(self):
        if len(self.demand_history) < 8:
            # print('not started calculating yet, current demand:', self.demand, 'month:', self.month)
            annual_feed_price_decrease = self.env.ann_feed_price_decrease
            new_price = self.price * np.power((1 - annual_feed_price_decrease), 1/12)

        else:
            demand_now = self.demand
            demand_six = self.demand_history[self.month - 6]

            delta_demand = demand_now/demand_six - 1

            feedstock_elasticity = self.price_elasticity
            # value from paper on the commit
            annual_feed_price_decrease = self.env.ann_feed_price_decrease

            new_price = self.price * (np.power((1 - annual_feed_price_decrease), 1/12) + delta_demand *
                                      feedstock_elasticity)
            # print(new_price)

        if self.random_switch:
            std_dev_ratio = 0.005
            deviation = np.float64(np.random.normal(0, (std_dev_ratio * new_price), None))
            self.price = new_price + deviation
        else:
            self.price = new_price
        return

    def get_price(self):
        return np.float64(self.price)

    def iterate_supplier(self):
        self.get_demand()
        self.set_price()
        self.month += 1
        self.env.parameter['bio_feedstock_price'].value = self.price
        return
