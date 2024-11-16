import pandas as pd
import matplotlib.pyplot as plt


class FutureCapacity:
    def __init__(self, amount: float, time: float, quarterly_cost: float):
        self.amount = amount
        self.time = time
        self.quarterly_cost = quarterly_cost


class Producer:
    def __init__(
        self,
        emission: float,
        capacity: dict[float, float] | int,
        cost: float,
        name: str,
        capital: float = 0,
        margin: float = 1,
    ):
        """Initializes a power plant.

        Args:
            emission: Emission in gCO2e/kWh
            capacity: Capacity in MW, either as a single value for constant
                      capacity or a dict with time as key and capacity in MW as
                      value. We expect the time to be in hours (24 hour clock) and
                      in constant intervals.
            cost: Cost in €/MWh
            name: Name of the power plant
            capital: Capital in €
            margin: How much the profit per MWh should be in €
        """
        self.emission = emission
        if isinstance(capacity, int):
            self.capacity = {0: capacity}
            self.initial_capacity = {0: capacity}
        elif isinstance(capacity, dict):
            self.capacity = capacity
            self.initial_capacity = capacity.copy()
        else:
            raise ValueError("Capacity should be either a dict or an int")
        self.cost = cost
        self.name = name
        self.capital = capital
        self.future_capacity: list[FutureCapacity] = []
        self.chunk_cost = 1_000_000  # €
        self.chunk_amount = 0.01  # Percentage of original capacity
        self.chunk_time = 4  # Quarters
        self.margin = margin
        self.quarterly_profits = []

    def decision_making(self):
        if self.capital > self.chunk_cost:
            cost_per_quarter = self.chunk_cost / self.chunk_time
            self.future_capacity.append(
                FutureCapacity(self.chunk_amount, self.chunk_time, cost_per_quarter)
            )

    def increase_capacity(self):
        if self.capital > self.chunk_cost:
            cost_per_quarter = self.chunk_cost / self.chunk_time
            self.future_capacity.append(
                FutureCapacity(self.chunk_amount, self.chunk_time, cost_per_quarter)
            )

    def decrease_capacity(self):
        if len(self.capacity) > 0:
            for k in self.capacity.keys():
                self.capacity[k] = max(self.capacity[k] - self.chunk_amount * self.initial_capacity[k], 0)

    def run_quarter(self, daily_production: float):
        daily_income = daily_production * self.cost
        daily_cost = daily_production * (self.cost - self.margin)
        daily_profit = daily_income - daily_cost
        quarterly_profit = daily_profit * 90
        # Update capital
        self.capital += quarterly_profit
        # Record quarterly profit
        self.quarterly_profits.append(quarterly_profit)
        # Update capacity
        self.update_capacity()
        # Make decision on increasing or decreasing capacity
        self.decision_making()

    def update_capacity(self):
        new_future_capacity = []
        for future_capacity in self.future_capacity:
            future_capacity.time -= 1
            self.capital -= future_capacity.quarterly_cost
            if future_capacity.time <= 0:
                self.new_capacity = {
                    k: v * future_capacity.amount
                    for k, v in self.initial_capacity.items()
                }
                self.capacity = add_dicts(self.capacity, self.new_capacity)
            else:
                new_future_capacity.append(future_capacity)
        self.future_capacity = new_future_capacity

    def capacity_f(self, time: float):
        """Returns the capacity of the power plant at a given time.

        Assumes that if the time falls between two time points, we return the
        capacity of the earlier time. E.g. If we have a capacity of 100 MW at
        12:00 and 200 MW at 13:00, the capacity at 12:30 will be 100 MW.

        Args:
            time: Time in hours (24 hour clock)

        Returns:
            capacity: Capacity in MW
        """
        times = sorted(self.capacity.keys())
        for t in reversed(times):
            if time >= t:
                return self.capacity[t]
        return 0

    def cost_f(self, time: float):
        return self.cost

    def emission_f(self, time: float):
        return self.emission

    def capital_f(self):
        return self.capital

    def __str__(self):
        return f"{self.name} - {self.capacity}, {self.cost}, {self.emission}, {self.capital}, {self.future_capacity}, {self.chunk_cost}, {self.chunk_amount}, {self.chunk_time}, {self.margin}"


class Coal(Producer):
    def __init__(
        self,
        emission: float = 820,
        capacity: float = 202_320,
        cost: float = 81.82,
        name: str = "Coal",
    ):
        super().__init__(emission, capacity, cost, name)


class Oil(Producer):
    def __init__(
        self,
        emission: float = 600,
        capacity: float = 22_000,
        cost: float = 150,
        name: str = "Oil",
    ):
        super().__init__(emission, capacity, cost, name)


class Gas(Producer):
    def __init__(
        self,
        emission: float = 490,
        capacity: float = 403_000,
        cost: float = 66.02,
        name: str = "Gas",
    ):
        super().__init__(emission, capacity, cost, name)


class Nuclear(Producer):
    def __init__(
        self,
        emission: float = 12,
        capacity: float = 147_774,
        cost: float = 64.16,
        name: str = "Nuclear",
    ):
        super().__init__(emission, capacity, cost, name)


class Hydro(Producer):
    def __init__(
        self,
        emission: float = 24,
        capacity: float = 230_000,
        cost: float = 66.95,
        name: str = "Hydro",
    ):
        super().__init__(emission, capacity, cost, name)


class Wind(Producer):
    def __init__(
        self,
        emission: float = 11,
        capacity: float = 255_000,
        cost: float = 46.49,
        name: str = "Wind",
    ):
        super().__init__(emission, capacity, cost, name)


class Solar(Producer):
    def __init__(
        self,
        emission: float = 48,
        capacity: float = 259_990,
        cost: float = 52.07,
        name: str = "Solar",
    ):
        super().__init__(emission, capacity, cost, name)


def add_dicts(dict1: dict[str, int], dict2: dict[str, int]) -> dict[str, int]:
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict


class Market:
    def run_day_interval_constant_demand(
        producers: list[Producer], demand: float, interval: int = 30
    ):
        """Runs the market for a day with constant demand

        Args:
            producers: List of producers
            demand: Demand in MW for the entire day
            interval: Time interval in minutes
        """
        intervals = 24 * 60 // interval
        demands = [demand] * intervals
        return Market.run_day_interval(producers, demands)

    def run_day_interval(producers: list[Producer], demands: list[float]):
        """Implements marginal pricing

        We assume a constant demand and production during our time step that is
        equal to the value at the start of the time step.

        Args:
            producers: List of producers
            demand: Demand in MW for each time step. Assumed to start at 00:00 and
                    be in constant intervals.
        """
        intervals = len(demands)
        interval_length = 24 / intervals

        total_cost = 0
        total_emission = 0
        total_production = {}
        interval_production = {}
        for interval in range(intervals):
            time = interval * interval_length
            demand = demands[interval]
            cost, emission, production = Market.run(
                producers, demand, time, interval_length
            )
            interval_production[interval] = production

            total_production = add_dicts(total_production, production)
            total_cost += cost
            total_emission += emission

        return total_cost, total_emission, total_production, interval_production

    def run(
        producers: list[Producer],
        demand: float,
        time: float = 0,
        interval_length: int = 24,
    ):
        """Implements marginal pricing

        We assume a constant demand and production during our time step that is
        equal to the value at the start of the time step.

        Args:
            producers: List of producers
            demand: Demand in MW at time
            time: Time at which the demand is calculated in hours (24 hour clock)
            interval_length: Time interval in hours

        Returns:
            total_cost: Total cost in €
            total_emission: Total emission in kgCO2e
            production: Dictionary with the name of the producer and the MWh produced
        """
        producers.sort(key=lambda x: x.cost_f(time))

        remaining_demand = demand
        total_emission = 0
        production = {producer.name: 0 for producer in producers}
        highest_cost = 0
        for producer in producers:
            if remaining_demand <= 0:
                break

            if producer.capacity_f(time) >= remaining_demand:
                total_emission += (
                    producer.emission_f(time) * remaining_demand * interval_length
                )
                production[producer.name] = remaining_demand * interval_length
                remaining_demand = 0
            else:
                total_emission += (
                    producer.emission_f(time)
                    * producer.capacity_f(time)
                    * interval_length
                )
                production[producer.name] = producer.capacity_f(time) * interval_length
                remaining_demand -= producer.capacity_f(time)

            highest_cost = producer.cost_f(time)

        total_cost = highest_cost * demand * interval_length
        return total_cost, total_emission, production


def plot_capacities(producers: list[Producer], demands: list[float]):
    times = sorted(
        set(time for producer in producers for time in producer.capacity.keys())
    )
    capacities = {
        producer.name: [producer.capacity_f(time) for time in times]
        for producer in producers
    }
    costs = {producer.name: producer.cost for producer in producers}

    sorted_producers = sorted(producers, key=lambda x: x.cost)
    sorted_names = [producer.name for producer in sorted_producers]

    fig, ax = plt.subplots()

    bottom = [0] * len(times)
    for name in sorted_names:
        ax.bar(times, capacities[name], bottom=bottom, label=name)
        bottom = [i + j for i, j in zip(bottom, capacities[name])]

    ax.plot(times, demands, label="Demand", color="red", linestyle="--")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Capacity (MW)")
    ax.legend()
    plt.show()


def plot_interval_production(interval_production: dict[int, dict[str, float]]):
    times = list(interval_production.keys())
    production = {producer: [] for producer in interval_production[0].keys()}

    for interval in times:
        for producer, amount in interval_production[interval].items():
            production[producer].append(amount)

    fig, ax = plt.subplots()

    bottom = [0] * len(times)
    for producer, amounts in production.items():
        ax.bar(times, amounts, bottom=bottom, label=producer)
        bottom = [i + j for i, j in zip(bottom, amounts)]

    ax.set_xlabel("Time (intervals)")
    ax.set_ylabel("Production (MWh)")
    ax.legend()
    plt.show()


if __name__ == "__main__":

    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    data = read_data("../data/data.csv")

    nuclear_capacity = data["Nuclear (GW)"] * 1_000
    nuclear_capacity_dict = nuclear_capacity.to_dict()
    nuclear = Nuclear(capacity=nuclear_capacity_dict)

    hydro_capacity = data["Hydro (GW)"] * 1_000
    hydro_capacity_dict = hydro_capacity.to_dict()
    hydro = Hydro(capacity=hydro_capacity_dict)

    wind_capacity = data["Wind (GW)"] * 1_000
    wind_capacity_dict = wind_capacity.to_dict()
    wind = Wind(capacity=wind_capacity_dict)

    solar_capacity = data["Solar (GW)"] * 1_000
    solar_capacity_dict = solar_capacity.to_dict()
    solar = Solar(capacity=solar_capacity_dict)

    demand_factor = 1 / 2
    demands = [value * 1_000 * demand_factor for value in data["Demand (GW)"]]

    plot_capacities([nuclear, hydro, wind, solar], demands)

    producers = [nuclear, hydro, wind, solar]

    number_of_quarters = 36
    for i in range(number_of_quarters):
        print(f"Quarter {i}")
        total_cost, total_emission, production, interval_production = (
            Market.run_day_interval(producers, demands)
        )
        print(f"Total cost: {total_cost} €")
        print(f"Total emission: {total_emission} kgCO2e")
        # print("Used producers:")
        for producer, amount in production.items():
            for p in producers:
                if p.name == producer:
                    # print(p)
                    p.run_quarter(amount)
            # print(f"{producer} - {amount} MWh")

        # plot_interval_production(interval_production)
