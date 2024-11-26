import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(results_file, parameters_file):
    """Load the results and parameters datasets."""
    results = pd.read_csv(results_file)
    parameters = pd.read_csv(parameters_file, usecols=["RES%", "NUC%"])
    return results, parameters


def compute_total_subsidies_and_costs(results):
    """
    Compute total subsidies and total cost to society.
    """
    total_subsidy_columns = [
        "Total Subsidy for Nuclear (€)",
        "Total Subsidy for Solar (€)",
        "Total Subsidy for Wind (€)",
        "Total Subsidy for Biomass (€)",
        "Total Subsidy for Gas (€)",
        "Total Subsidy for Hydro (€)",
        "Total Subsidy for Coal (€)",
    ]
    # Sum subsidies across energy types for each row
    results["Total Subsidy (€)"] = results[total_subsidy_columns].sum(axis=1)
    # Compute total cost to society
    results["Total Cost to Society (€)"] = (
        results["Total Cost (€)"] + results["Total Subsidy (€)"]
    )
    return results


def get_total_values(results, parameters, values):
    """
    Compute total values by summing over all quarters and merge with parameters.
    """
    total_values = results.groupby("Dataset Row")[values].sum().reset_index()
    total_values_with_parameters = total_values.merge(
        parameters, left_on="Dataset Row", right_index=True
    ).drop(columns=["Dataset Row"])
    return total_values_with_parameters


def get_values_for_quarter(results, parameters, quarter):
    """
    Filter values for a specific quarter and merge with parameters.
    """
    filtered_results = results[results["Quarter"] == quarter]
    values_for_quarter = filtered_results.merge(
        parameters, left_on="Dataset Row", right_index=True
    ).drop(columns=["Dataset Row", "Quarter"])
    return values_for_quarter


def prepare_heatmap_data(values_data, values):
    """
    Prepare data for the heatmap by rounding percentages and creating a pivot table.
    """
    values_data["NUC%"] = values_data["NUC%"].round(2)
    values_data["RES%"] = values_data["RES%"].round(2)
    heatmap_data = values_data.pivot_table(index="NUC%", columns="RES%", values=values)
    return heatmap_data


def plot_heatmap(heatmap_data, title):
    """
    Plot a heatmap from the prepared data with (0,0) in the bottom left.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=False,
        fmt=".1f",
        cmap="flare",
        cbar_kws={"orientation": "vertical"},
    )

    plt.gca().invert_yaxis()  # Invert the y-axis to place (0,0) in the bottom left

    plt.title(title)
    plt.xlabel("RES%")
    plt.ylabel("NUC%")
    plt.show()


def plot_values_over_time(results, parameters, res_value, nuc_value, values):
    """
    Plot total values over time for a specific combination of RES% and NUC%.

    Parameters:
        results (DataFrame): Results dataset.
        parameters (DataFrame): Parameters dataset.
        res_value (float): Desired RES% value.
        nuc_value (float): Desired NUC% value.
    """
    # Merge results with parameters to associate values with RES% and NUC%
    merged_data = results.merge(parameters, left_on="Dataset Row", right_index=True)

    # Filter data for the specific RES% and NUC% values
    filtered_data = merged_data[
        (merged_data["RES%"].round(2) == round(res_value, 2))
        & (merged_data["NUC%"].round(2) == round(nuc_value, 2))
    ]

    # Aggregate total cost to society by quarter for the specific parameter combination
    values_over_time = filtered_data.groupby("Quarter")[values].sum().reset_index()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        values_over_time["Quarter"],
        values_over_time[values],
        marker="o",
    )
    plt.title(f"{values} Over Time\n(RES%={res_value}, NUC%={nuc_value})")
    plt.xlabel("Quarter")
    plt.ylabel(values)
    plt.grid(True)
    plt.show()


def plot_multiple_metrics_over_time_single_plot(
    results, parameters, parameter_combinations, metrics
):
    """
    Plot multiple metrics (e.g., Capacity and Production) over time for multiple parameter combinations in a single plot.

    Parameters:
        results (DataFrame): Results dataset.
        parameters (DataFrame): Parameters dataset.
        parameter_combinations (list of tuples): List of (RES%, NUC%) combinations.
        metrics (list of str): List of column names for the metrics to visualize.
    """
    merged_data = results.merge(parameters, left_on="Dataset Row", right_index=True)

    plt.figure(figsize=(12, 8))

    for metric in metrics:
        for res_value, nuc_value in parameter_combinations:
            # Filter data for the specific RES% and NUC% combination
            filtered_data = merged_data[
                (merged_data["RES%"].round(2) == round(res_value, 2))
                & (merged_data["NUC%"].round(2) == round(nuc_value, 2))
            ]

            # Aggregate metric by quarter
            metric_over_time = (
                filtered_data.groupby("Quarter")[metric].sum().reset_index()
            )

            # Plot on the same figure
            plt.plot(
                metric_over_time["Quarter"],
                metric_over_time[metric],
                marker="o",
                label=f"{metric} (RES%={res_value}, NUC%={nuc_value})",
            )

    plt.title("Metrics Over Time for Multiple Parameter Combinations")
    plt.xlabel("Quarter")
    plt.ylabel("Value")
    plt.legend(
        title="Metrics & Parameter Combinations",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multiple_metrics_over_time_subplots(
    results, parameters, parameter_combinations, metrics
):
    """
    Plot multiple metrics (e.g., Capacity and Production) over time for multiple parameter combinations in subplots.

    Parameters:
        results (DataFrame): Results dataset.
        parameters (DataFrame): Parameters dataset.
        parameter_combinations (list of tuples): List of (RES%, NUC%) combinations.
        metrics (list of str): List of column names for the metrics to visualize.
    """
    num_combinations = len(parameter_combinations)
    rows = (num_combinations + 1) // 2  # Arrange subplots in 2 columns
    cols = 2 if num_combinations > 1 else 1

    fig, axes = plt.subplots(
        rows, cols, figsize=(15, rows * 5), sharex=True, sharey=True
    )
    axes = axes.flatten() if num_combinations > 1 else [axes]

    merged_data = results.merge(parameters, left_on="Dataset Row", right_index=True)

    for i, (res_value, nuc_value) in enumerate(parameter_combinations):
        # Filter data for the specific RES% and NUC% combination
        filtered_data = merged_data[
            (merged_data["RES%"].round(2) == round(res_value, 2))
            & (merged_data["NUC%"].round(2) == round(nuc_value, 2))
        ]

        # Plot each metric for the current parameter combination
        for metric in metrics:
            # Aggregate metric by quarter
            metric_over_time = (
                filtered_data.groupby("Quarter")[metric].sum().reset_index()
            )

            # Plot on the current subplot
            axes[i].plot(
                metric_over_time["Quarter"],
                metric_over_time[metric],
                marker="o",
                label=f"{metric}",
            )

        axes[i].set_title(f"RES%={res_value}, NUC%={nuc_value}")
        axes[i].set_xlabel("Quarter")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
        axes[i].legend(title="Metrics", loc="upper left")

    # Hide empty subplots if parameter_combinations < rows * cols
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# Main workflow
if __name__ == "__main__":
    # Load the data
    results, parameters = load_data("simulation_results.csv", "subsidies.csv")

    # Compute total subsidies and total cost to society
    results = compute_total_subsidies_and_costs(results)

    quarter = 60
    quarter_values = get_values_for_quarter(results, parameters, quarter)

    # Total emissions heatmap
    total_emissions = get_total_values(
        results, parameters, "Total CO2 Emission (kgCO2e)"
    )
    total_heatmap_data = prepare_heatmap_data(
        total_emissions, "Total CO2 Emission (kgCO2e)"
    )
    plot_heatmap(
        total_heatmap_data, "Heatmap of Total CO2 Emission (kgCO2e) Across All Quarters"
    )

    quarter_heatmap_data = prepare_heatmap_data(
        quarter_values, "Total CO2 Emission (kgCO2e)"
    )
    plot_heatmap(
        quarter_heatmap_data, f"Heatmap of CO2 Emission (kgCO2e) for Quarter {quarter}"
    )

    # Visualize CO2 emissions over time for a specific parameter combination
    plot_values_over_time(
        results,
        parameters,
        res_value=0,
        nuc_value=0,
        values="Total CO2 Emission (kgCO2e)",
    )

    # Total costs heatmap
    total_costs = get_total_values(results, parameters, "Total Cost to Society (€)")
    total_costs_heatmap_data = prepare_heatmap_data(
        total_costs, "Total Cost to Society (€)"
    )
    plot_heatmap(
        total_costs_heatmap_data,
        "Heatmap of Total Cost to Society (€) Across All Quarters",
    )

    quarter_costs_heatmap_data = prepare_heatmap_data(
        quarter_values, "Total Cost to Society (€)"
    )
    plot_heatmap(
        quarter_costs_heatmap_data,
        f"Heatmap of Total Cost to Society (€) for Quarter {quarter}",
    )

    # Visualize total costs over time for a specific parameter combination
    plot_values_over_time(
        results,
        parameters,
        res_value=0,
        nuc_value=0,
        values="Total Cost to Society (€)",
    )
    plot_values_over_time(
        results,
        parameters,
        res_value=0,
        nuc_value=0,
        values="Total Cost to Society (€)",
    )

    # Define parameter combinations (list of (RES%, NUC%))
    parameter_combinations = [
        (0.30, 0.70),  # RES%=30, NUC%=70
        (0.50, 0.50),  # RES%=50, NUC%=50
        (0.70, 0.30),  # RES%=70, NUC%=30
    ]

    # Define cost metrics
    cost_metrics = ["Total Cost (€)", "Total Subsidy (€)", "Total Cost to Society (€)"]

    # Single plot visualization for costs
    plot_multiple_metrics_over_time_single_plot(
        results, parameters, parameter_combinations, cost_metrics
    )

    # Subplots visualization for costs
    plot_multiple_metrics_over_time_subplots(
        results, parameters, parameter_combinations, cost_metrics
    )

    # Continue with your existing code for other metrics...

    capital = [
        "Nuclear Capital (€)",
        "Solar Capital (€)",
        "Wind Capital (€)",
        "Biomass Capital (€)",
        "Gas Capital (€)",
        "Hydro Capital (€)",
        "Coal Capital (€)",
    ]

    buildings = [
        "Nuclear plants building",
        "Solar plants building",
        "Wind plants building",
        "Biomass plants building",
        "Gas plants building",
        "Hydro plants building",
        "Coal plants building",
    ]

    capacities = [
        "Nuclear Capacity (MW)",
        "Solar Capacity (MW)",
        "Wind Capacity (MW)",
        "Biomass Capacity (MW)",
        "Gas Capacity (MW)",
        "Hydro Capacity (MW)",
        "Coal Capacity (MW)",
    ]

    productions = [
        "Nuclear Production (MWh)",
        "Solar Production (MWh)",
        "Wind Production (MWh)",
        "Biomass Production (MWh)",
        "Gas Production (MWh)",
        "Hydro Production (MWh)",
        "Coal Production (MWh)",
    ]

    nuclear_metrics = [
        "Nuclear Capacity (MW)",
        "Nuclear Production (MWh)",
    ]

    # Plotting other metrics as before...
    plot_multiple_metrics_over_time_single_plot(
        results, parameters, [(0, 0)], capacities
    )
    plot_multiple_metrics_over_time_single_plot(
        results, parameters, [(0, 0)], productions
    )
    plot_multiple_metrics_over_time_single_plot(
        results, parameters, [(0, 0)], nuclear_metrics
    )
    plot_multiple_metrics_over_time_single_plot(results, parameters, [(0, 0)], capital)
    plot_multiple_metrics_over_time_single_plot(
        results, parameters, [(0, 0)], buildings
    )

    # Subplots visualization
    plot_multiple_metrics_over_time_subplots(
        results, parameters, parameter_combinations, capacities
    )
    plot_multiple_metrics_over_time_subplots(
        results, parameters, parameter_combinations, productions
    )
    plot_multiple_metrics_over_time_subplots(
        results, parameters, parameter_combinations, nuclear_metrics
    )
    plot_multiple_metrics_over_time_subplots(
        results, parameters, parameter_combinations, capital
    )
    plot_multiple_metrics_over_time_subplots(
        results, parameters, parameter_combinations, buildings
    )
