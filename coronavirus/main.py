import pandas as pd
from extraction import (
    extract_process_data,
)
from visualization import plot_data, plot_data_by_country

(
    confirmed_by_country,
    deaths_by_country,
    recovered_by_country,
    recovered_or_passed_by_country,
    infected_by_country,
) = extract_process_data()


plot_data_by_country(deaths_by_country, "Germany")
