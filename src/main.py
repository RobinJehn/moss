import pandas as pd
import matplotlib.pyplot as plt
import csv


def log(msg: str):
    should_print = False
    if should_print:
        print(msg)


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
        fixed_om: float = 0,  # Fixed O&M cost per MW of capacity (€ per quarter)
        variable_om: float = 0,  # Variable O&M cost per MWh produced (€ per MWh)
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
            fixed_om: Fixed O&M cost per MW per quarter
            variable_om: Variable O&M cost per MWh produced
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
        self.variable_om = variable_om  # Added variable O&M
        self.fixed_om = fixed_om  # Added fixed O&M
        self.name = name
        self.capital = capital
        self.future_capacity: list[FutureCapacity] = []
        self.chunk_cost = 0  # €
        self.chunk_amount = 0  # Percentage of original capacity
        self.chunk_time = 0  # Quarters
        self.quarterly_profits = []
        self.planned_investment = 0
        self.time_per_section = 24 / len(self.capacity.keys())

    def reset(self):
        self.capital = 0
        self.capacity = self.initial_capacity.copy()
        self.future_capacity = []
        self.quarterly_profits = []
        self.planned_investment = 0

    def decision_making(self):
        # Check if we have enough quarterly profit data to make decisions
        if len(self.quarterly_profits) >= 3:
            # Check the last three quarters' profits
            last_three_profits = self.quarterly_profits[-3:]

            # If profits decreased for three consecutive quarters, decommission capacity
            if all(last_three_profits[i] > last_three_profits[i + 1] for i in range(2)):
                self.decrease_capacity()

        # Check if we have at least one quarter of profit data to determine profit increase
        if len(self.quarterly_profits) >= 2:
            # Compare the most recent two quarters to determine if profits have increased
            if self.quarterly_profits[-1] >= self.quarterly_profits[-2]:
                self.increase_capacity()

    def increase_capacity(self):
        if self.chunk_cost == 0:
            return
        min_remaining = 0  # 3 * self.chunk_cost
        available_capital = max(
            self.capital - self.planned_investment - min_remaining, 0
        )
        number_to_add = min(available_capital // self.chunk_cost, 20)

        cost_per_quarter = self.chunk_cost / self.chunk_time
        self.planned_investment += self.chunk_cost * number_to_add
        for _ in range(int(number_to_add)):
            self.future_capacity.append(
                FutureCapacity(self.chunk_amount, self.chunk_time, cost_per_quarter)
            )

    def decrease_capacity(self):
        if len(self.future_capacity) > 0:
            self.future_capacity.pop(0)

        if self.quarterly_profits[-1] > self.chunk_cost:
            return

        if len(self.capacity) > 0:
            for k in self.capacity.keys():
                self.capacity[k] = max(
                    self.capacity[k] - self.chunk_amount * self.initial_capacity[k],
                    0,
                )

    def run_quarter(
        self,
        daily_production: float,
        marginal_price: float,
        quarterly_subsidies: float = 0,
    ):
        daily_income = daily_production * marginal_price

        # Calculate daily cost based on producer's own cost of production
        daily_cost = daily_production * (self.cost + self.variable_om)
        daily_profit = daily_income - daily_cost
        quarterly_profit = daily_profit * 90

        # Substract/add fixed values
        fixed_om_cost = sum(self.capacity.values()) * self.fixed_om
        net_quarterly_profit = quarterly_profit - fixed_om_cost + quarterly_subsidies

        # Record quarterly profit
        self.capital += net_quarterly_profit
        self.quarterly_profits.append(net_quarterly_profit)

        # Update capacity
        self.update_capacity()

        # Make decision on increasing or decreasing capacity
        self.decision_making()

    def update_capacity(self):
        new_future_capacity = []
        for future_capacity in self.future_capacity:
            future_capacity.time -= 1
            self.capital -= future_capacity.quarterly_cost
            self.planned_investment -= future_capacity.quarterly_cost
            if future_capacity.time <= 0:
                self.new_capacity = {
                    k: v * future_capacity.amount
                    for k, v in self.initial_capacity.items()
                }
                self.capacity = add_dicts(self.capacity, self.new_capacity)
            else:
                new_future_capacity.append(future_capacity)
        self.future_capacity = new_future_capacity

    def capacity_f(self, time: float) -> float:
        """Returns the capacity of the power plant at a given time.

        Assumes that if the time falls between two time points, we return the
        capacity of the earlier time. E.g. If we have a capacity of 100 MW at
        12:00 and 200 MW at 13:00, the capacity at 12:30 will be 100 MW.

        Args:
            time: Time in hours (24 hour clock)

        Returns:
            capacity: Capacity in MW
        """
        section = int(time // self.time_per_section)
        return self.capacity[section]

    def cost_f(self, time: float) -> float:
        return self.cost

    def emission_f(self, time: float) -> float:
        return self.emission

    def capital_f(self) -> float:
        return self.capital

    def __str__(self) -> str:
        return f"{self.name} - {self.capacity}, {self.cost}, {self.emission}, {self.capital}, {self.future_capacity}, {self.chunk_cost}, {self.chunk_amount}, {self.chunk_time}"


class SubsidySimulation:
    def __init__(self, subsidies_df: pd.DataFrame):
        self.subsidies_df = subsidies_df

    def simulate_subsidies(
        self, idx: int, producer: Producer, production_mwh: float
    ) -> float:
        """
        Calculates how much subsidies the producer gets for the given quarter.

        Args:
            idx: Which subsidy parameters to use
            producer: Which producer to subsidies
            production_mwh: How many mwh the producer produced for this quarter.

        Returns:
            Amount of subsidies
        """

        # Ensure the quarter is within bounds of the subsidies data
        if idx < len(self.subsidies_df):
            # Get the subsidies for the current quarter
            subsidy_data = self.subsidies_df.iloc[idx]

            # Determine the subsidy per MWh based on the producer type
            if isinstance(producer, Wind):
                subsidy_per_mwh = subsidy_data["Wind"]
            elif isinstance(producer, Solar):
                subsidy_per_mwh = subsidy_data["Solar"]
            elif isinstance(producer, Gas):
                subsidy_per_mwh = subsidy_data["Gas"]
            elif isinstance(producer, Hydro):
                subsidy_per_mwh = subsidy_data["Hydro"]
            elif isinstance(producer, Coal):
                subsidy_per_mwh = subsidy_data["Coal"]
            elif isinstance(producer, Nuclear):
                subsidy_per_mwh = subsidy_data["Nuclear"]
            else:
                subsidy_per_mwh = 0

            # Calculate total subsidy based on production (subsidy per MWh * MWh produced)
            total_subsidy = subsidy_per_mwh * production_mwh

            log(
                f"Subsidy for {producer.name} in Quarter {quarter + 1}: {total_subsidy:.2f} €"
            )

            # Return the calculated subsidy value
            return total_subsidy

        else:
            log(f"Warning: Quarter {quarter} exceeds available subsidy data.")
            return 0


class Coal(Producer):
    def __init__(
        self,
        emission: float = 820,
        capacity: float = 202_320,
        cost: float = 81.82,
        name: str = "Coal",
        fixed_om: float = 92.8,
        variable_om: float = 3.0,
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)
        self.chunk_cost = 1_600_000_000  # 1600 million EUR
        self.chunk_amount = 0.01  # 1% of initial capacity
        self.chunk_time = 20  # Expansion time in quarters


class Gas(Producer):
    def __init__(
        self,
        emission: float = 490,
        capacity: float = 403_000,
        cost: float = 66.02,
        name: str = "Gas",
        fixed_om: float = 18.4,  # Fixed O&M (€ per MW per quarter)
        variable_om: float = 3.0,  # Variable O&M (€ per MWh)
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)
        self.chunk_cost = 50_000_000  # 500 million EUR
        self.chunk_amount = 0.001  # 1% of initial capacity
        self.chunk_time = 10  # Expansion time in quarters


class Nuclear(Producer):
    def __init__(
        self,
        emission: float = 12,
        capacity: float = 147_774,
        cost: float = 64.16,
        name: str = "Nuclear",
        fixed_om: float = 157.5,
        variable_om: float = 6.4,
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)
        self.chunk_cost = 8_000_000_000  # 8000 million EUR
        self.chunk_amount = 0.02  # 2% of initial capacity
        self.chunk_time = 32  # Expansion time in quarters


class Biomass(Producer):
    def __init__(
        self,
        emission: float = 600,
        capacity: float = 11_000,
        cost: float = 150,
        name: str = "Biomass",
        fixed_om: float = 92.8,
        variable_om: float = 3.0,
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)


class Hydro(Producer):
    def __init__(
        self,
        emission: float = 24,
        capacity: float = 230_000,
        cost: float = 66.95,
        name: str = "Hydro",
        fixed_om: float = 32.2,
        variable_om: float = 0.0,
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)


class Wind(Producer):
    def __init__(
        self,
        emission: float = 11,
        capacity: float = 255_000,
        cost: float = 46.49,
        name: str = "Wind",
        fixed_om: float = 16,
        variable_om: float = 0.02,
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)
        self.chunk_cost = 125_000_000  # 125 million EUR
        self.chunk_amount = 0.003  # 0.3% of initial capacity
        self.chunk_time = 6  # Expansion time in quarters


class Solar(Producer):
    def __init__(
        self,
        emission: float = 48,
        capacity: float = 259_990,
        cost: float = 52.07,
        name: str = "Solar",
        fixed_om: float = 18.8,
        variable_om: float = 0.0,
    ):
        super().__init__(emission, capacity, cost, name, fixed_om, variable_om)
        self.chunk_cost = 125_000_000  # 125 million EUR
        self.chunk_amount = 0.002  # 0.2% of initial capacity
        self.chunk_time = 6  # Expansion time in quarters


def add_dicts(dict1: dict[str, float], dict2: dict[str, float]) -> dict[str, float]:
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
    ) -> tuple[float, float, dict[str, float], dict[int, dict[str, float]]]:
        """Runs the market for a day with constant demand

        Args:
            producers: List of producers
            demand: Demand in MW for the entire day
            interval: Time interval in minutes

         Returns:
            total_cost: Total cost in €
            total_emission: Total emissions in kgCO2e
            total_production: Dictionary with the name of the producer and the MWh produced
            interval_production: Dictionary with the interval as key and the production as value
        """
        intervals = 24 * 60 // interval
        demands = [demand] * intervals
        return Market.run_day_interval(producers, demands)

    def run_day_interval(
        producers: list[Producer], demands: list[float]
    ) -> tuple[float, float, dict[str, float], dict[int, dict[str, float]]]:
        """Implements marginal pricing

        We assume a constant demand and production during our time step that is
        equal to the value at the start of the time step.

        Args:
            producers: List of producers
            demand: Demand in MW for each time step. Assumed to start at 00:00 and
                    be in constant intervals.

        Returns:
            total_cost: Total cost in €
            total_emission: Total emissions in kgCO2e
            total_production: Dictionary with the name of the producer and the MWh produced
            interval_production: Dictionary with the interval as key and the production as value
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
    ) -> tuple[float, float, dict[str, float]]:
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


def print_producer_metrics(producers, label):
    log(f"\n--- {label} ---")
    for producer in producers:
        profit = producer.quarterly_profits[-1] if producer.quarterly_profits else 0
        log(
            f"{producer.name}: Capacity = {sum(producer.capacity.values()):.2f} MW, Capital = {producer.capital:.2f} €, Profit This Quarter = {profit:.2f} €"
        )

        # Print a warning if capital or profit is negative
        if profit < 0:
            log(
                f"WARNING: {producer.name} has negative profit of {profit:.2f} € in this quarter."
            )
        if producer.capital < 0:
            log(
                f"WARNING: {producer.name} has negative capital of {producer.capital:.2f} €, indicating financial trouble."
            )


if __name__ == "__main__":
    output_csv_path = "simulation_results.csv"
    with open(output_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(
            [
                "Dataset Row",
                "Quarter",
                "Total CO2 Emission (kgCO2e)",
                "Total Cost (€)",
                "Total Subsidy for Nuclear (€)",
                "Total Subsidy for Solar (€)",
                "Total Subsidy for Wind (€)",
                "Total Subsidy for Biomass (€)",
                "Total Subsidy for Gas (€)",
                "Total Subsidy for Hydro (€)",
                "Total Subsidy for Coal (€)",
                "Nuclear Capacity (MW)",
                "Solar Capacity (MW)",
                "Wind Capacity (MW)",
                "Biomass Capacity (MW)",
                "Gas Capacity (MW)",
                "Hydro Capacity (MW)",
                "Coal Capacity (MW)",
                "Nuclear Production (MWh)",
                "Solar Production (MWh)",
                "Wind Production (MWh)",
                "Biomass Production (MWh)",
                "Gas Production (MWh)",
                "Hydro Production (MWh)",
                "Coal Production (MWh)",
                "Nuclear Capital (€)",
                "Solar Capital (€)",
                "Wind Capital (€)",
                "Biomass Capital (€)",
                "Gas Capital (€)",
                "Hydro Capital (€)",
                "Coal Capital (€)",
                "Nuclear plants building",
                "Solar plants building",
                "Wind plants building",
                "Biomass plants building",
                "Gas plants building",
                "Hydro plants building",
                "Coal plants building",
            ]
        )

    subsidies_data = pd.read_csv("subsidies.csv")
    subsidy_simulator = SubsidySimulation(subsidies_data)

    data = pd.read_csv("../data/data.csv")

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

    coal_capacity = data["Coal (GW)"] * 1_000
    coal_capacity_dict = coal_capacity.to_dict()
    coal = Coal(capacity=coal_capacity_dict)

    gas_capacity = data["Gas (GW)"] * 1_000
    gas_capacity_dict = gas_capacity.to_dict()
    gas = Gas(capacity=gas_capacity_dict)

    biomass_capacity = data["Biomass (GW)"] * 1_000
    biomass_capacity_dict = biomass_capacity.to_dict()
    biomass = Biomass(capacity=biomass_capacity_dict)

    demand_factor = 1
    demands = [value * 1_000 * demand_factor for value in data["Demand (GW)"]]

    producers: list[Producer] = [nuclear, hydro, wind, solar, coal, gas, biomass]
    producers.sort(key=lambda x: x.cost_f(0))
    name_to_index = {producer.name: idx for idx, producer in enumerate(producers)}
    # plot_capacities(producers, demands)

    for idx in range(len(subsidies_data)):
        log(f"\n--- Running Simulation for Subsidy Data Row {idx+1} ---")

        for producer in producers:
            producer.reset()

        previous_capacity = {
            producer.name: sum(producer.capacity.values()) for producer in producers
        }

        for quarter in range(60):
            log(f"\nQuarter {quarter + 1}")

            (
                daily_total_cost,
                daily_total_emission,
                daily_production,
                daily_interval_production,
            ) = Market.run_day_interval(producers, demands)
            daily_total_demand = sum(demands)
            marginal_price = daily_total_cost / daily_total_demand

            log(f"Total daily cost: {daily_total_cost} €")
            log(f"Total daily emission: {daily_total_emission} kgCO2e")

            quarterly_subsidies = {
                "Biomass": 0,
                "Nuclear": 0,
                "Solar": 0,
                "Wind": 0,
                "Biomass": 0,
                "Gas": 0,
                "Hydro": 0,
                "Coal": 0,
            }
            for producer, daily_amount in daily_production.items():
                p_idx = name_to_index[producer]
                p = producers[p_idx]
                quarterly_amount = daily_amount * 90
                log(
                    f"Production for {p.name} in Quarter {quarter+1}: {quarterly_amount:.2f} MWh"
                )

                subsidy = subsidy_simulator.simulate_subsidies(idx, p, quarterly_amount)
                quarterly_subsidies[p.name] += subsidy
                p.run_quarter(
                    daily_amount,
                    marginal_price=marginal_price,
                    quarterly_subsidies=subsidy,
                )
            quarterly_emissions = daily_total_emission * 90
            quarterly_costs = daily_total_cost * 90
            with open(output_csv_path, mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        idx,
                        quarter + 1,
                        quarterly_emissions,
                        quarterly_costs,
                        quarterly_subsidies["Nuclear"],
                        quarterly_subsidies["Solar"],
                        quarterly_subsidies["Wind"],
                        quarterly_subsidies["Biomass"],
                        quarterly_subsidies["Gas"],
                        quarterly_subsidies["Hydro"],
                        quarterly_subsidies["Coal"],
                        previous_capacity["Nuclear"],
                        previous_capacity["Solar"],
                        previous_capacity["Wind"],
                        previous_capacity["Biomass"],
                        previous_capacity["Gas"],
                        previous_capacity["Hydro"],
                        previous_capacity["Coal"],
                        daily_production["Nuclear"],
                        daily_production["Solar"],
                        daily_production["Wind"],
                        daily_production["Biomass"],
                        daily_production["Gas"],
                        daily_production["Hydro"],
                        daily_production["Coal"],
                        nuclear.capital,
                        solar.capital,
                        wind.capital,
                        biomass.capital,
                        gas.capital,
                        hydro.capital,
                        coal.capital,
                        len(nuclear.future_capacity),
                        len(solar.future_capacity),
                        len(wind.future_capacity),
                        len(biomass.future_capacity),
                        len(gas.future_capacity),
                        len(hydro.future_capacity),
                        len(coal.future_capacity),
                    ]
                )

            previous_capacity = {
                producer.name: sum(producer.capacity.values()) for producer in producers
            }

            # print_producer_metrics(producers, "After Applying Subsidies")
        # plot_interval_production(daily_interval_production)
        # plot_capacities(producers, demands)
