import pandas as pd
from matplotlib import pyplot as plt


def plot_results(results_df: pd.DataFrame) -> None:
    """
    Plots the results from a DataFrame as a line plot.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the data to plot.
                                   The index should represent the x-axis (e.g., time).
    """
    # Line plot of the DataFrame
    ax = results_df.plot(kind='line', figsize=(20, 6))

    # Labeling and formatting
    plt.xlabel('Time')
    plt.ylabel('System Output (kW)')
    plt.title('System Output Over Time')
    plt.legend(title='Variables', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid(visible=True, linestyle='--', linewidth=0.5)

    # Display the plot
    plt.tight_layout()
    plt.show()


def analyze_production(
        results_df: pd.DataFrame, frequency
):
    max_production = results_df['inverter_ac'].max()
    max_production_time = results_df['inverter_ac'].idxmax()
    production_threshold = 0
    production_start = results_df[results_df['inverter_ac'] > production_threshold].index.min()
    production_end = results_df[results_df['inverter_ac'] > production_threshold].index.max()

    print(f"Total AC energy: {results_df['inverter_ac'].sum() * frequency:.2f} kWh")
    print(f"Max production: {max_production:.2f} kW at {max_production_time}")
    print(f"Production starts at: {production_start}, ends at: {production_end}")
    plot_results(results_df)

    return max_production, max_production_time, production_start, production_end
