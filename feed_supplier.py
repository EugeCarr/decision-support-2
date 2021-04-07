from agent import Agent
import numpy as np
import math


class Supplier(Agent):
    """So the feedstock supplier will have 1 job. To continually create resource X, and supply whatever of Resource X
    there is to the Manufacturers, and set set prices for Resource X based on the availability of said resource. The
    idea is that the resource will be renewable, for example trees. This means that changes to amount in trees, will
    take 10 years to be realised.

    It needs to have: a starting price of the resource per kilo, a initial amount of that resource, maximum amount of
    that resource, a current growth rate of the resource stock, A timer for the resource to be replenished for new
    growth, A pricing policy that determines the new price based on the current demand to stock ratio.

    Also needs to communicate to PET manufacturers that there is no more growth in reserves once the max amount has been
    reached. After this point the companies will have to consider resource scarcity increasing the price of feedstock.

    Also needs to communicate that the price is changing when that happens"""

    def __init__(self, name, sim_time, env, start_price, initial_amount, max_amount, ann_growth, proportion_feedstock,
                 replenish_time, sensitivity):
        super().__init__(name, sim_time, env)
        assert type(start_price) == float, ("start_price should be a float, not a", type(start_price))
        assert type(initial_amount) == float, ("initial_amount should be a float, not a", type(initial_amount))
        assert type(max_amount) == float, ("max_amount should be a float, not a", type(max_amount))
        assert type(ann_growth) == float and 0.0 > ann_growth > 1.0, ("growth should be a float, not a",
                                                                        type(ann_growth))
        assert type(replenish_time) == int, ("replenish_time should be a int, not a", type(replenish_time))
        assert type(proportion_feedstock) == float and 0.0 > abs(proportion_feedstock) > 1.0, (
                            "proportion_feedstock should be ratio between 0 and 1, not a",type(proportion_feedstock),
                            "value", proportion_feedstock)
        assert type(sensitivity) == float, ("sensitivity should be a float, not a", type(sensitivity))

        self.start_price = start_price
        self.nat_stock = initial_amount

        self.maximum = max_amount
        self.growth = ann_growth

        if replenish_time == None:
            self.replenish_time = 10
        else:
            self.replenish_time = replenish_time

        self.price = start_price
        self.prop_feedstock = proportion_feedstock
        self.reserves = self.nat_stock * self.prop_feedstock
        self.reserve_history = []

        self.demand = np.float()
        self.demand_baseline = np.float()
        self.demand_history = []
        self.ratio_baseline = np.float()

        self.sensitivity = sensitivity

        self.planting = False
        self.new_resource = np.float(0.0)
        self.resource_timer = 0

        self.inc = False
        self.prop_inc = 0.0
        self.prop_inc_timer = 0

        self.random_switch = False

        return

    def calculate_reserves(self):
        self.update_nat_stock()
        # this just ensures that any new resource planted is counted in the calculation
        self.reserves = self.prop_feedstock * self.nat_stock
        self.reserve_history.append(self.reserves)
        return

    def update_nat_stock(self):
        if self.nat_stock < self.maximum:
            self.nat_stock *= 1 + math.pow(self.growth, 1 / 12)
        return

    def change_growth(self, new_growth):
        assert float == type(new_growth) and 0.0 > abs(new_growth) > 1.0, ("new_growth should be a float, not a",
                                                                           type(new_growth))
        self.growth = new_growth
        return

    def get_demand(self, demand):
        assert type(demand) == float, ("demand should be a float, not a", type(demand))
        self.demand = demand
        if len(self.demand_history) == 0:
            self.demand_baseline = np.float(demand)

        self.demand_history.append(demand)
        return

    def set_price(self):
        ratio = self.demand / self.reserves

        if len(self.demand_history) == 0:
            self.ratio_baseline = ratio
        else:
            diff = (ratio - self.ratio_baseline)/ self.ratio_baseline
        #     calculates the difference in supply demand ratio between now and the beginning
            new_price = self.price * (1 + diff * self.sensitivity)
        #     sensitivity allows you to change the percentage by which the price changes in response to supply and
        #     demand.
            if self.random_switch:
                std_dev = 0.01
                deviation = np.float64(np.random.normal(0, std_dev, None))
                self.price = new_price + deviation
            else:
                self.price = new_price
        return

    def increase_resource(self, proportion_added):
        self.planting = True
        #     this is basically if the regulator decides to plant more trees
        assert type(proportion_added) == float and 0.0 < proportion_added < 1.0,("input must be a float between 0 and 1",
                                                                                  type(proportion_added))
        self.new_resource = self.nat_stock * proportion_added
        self.resource_timer = self.replenish_time
#       now there's a timer before the stocks get boosted
        return

    def increment_plant_resource(self):
        if not self.planting:
            return
        self.resource_timer -= 1
        if self.resource_timer == 0:
            self.nat_stock += self.new_resource
            print("Supplier:", self.name, " has just increased stock by:", self.new_resource, "on month:", self.month)
            self.new_resource = 0.0
            self.planting = False
        return

    def increase_proportion(self, increment, phase_in):
        assert type(increment) == float and 0.0 < increment < 1.0, ("input must be a float between 0 and 1",
                                                                    type(increment))
        assert type(phase_in) == int , ("phase_in month timer must be an integer", type(phase_in))

        self.prop_inc = increment / phase_in
        self.prop_inc_timer = phase_in
        self.inc = True
        return

    def increment_proportion(self):
        if not self.inc:
            return
        if self.prop_inc_timer > 0:
            self.prop_feedstock += self.prop_inc
            self.prop_inc_timer -= 1

            if self.prop_inc_timer == 0:
                self.inc = False
            return

    def iterate_supplier(self, demand):
        self.get_demand(demand)
        self.calculate_reserves()
        self.set_price()
        self.increment_proportion()
        self.increment_proportion()
        self.month += 1
        return



