import pandas as pd
from matplotlib import pyplot as plt

from coronavirus.extraction import (
    extract_process_data,
    data_gouv_vue_ensemble,
    data_gouv_taux_incidence,
    data_gouv_hospital,
    data_gouv_vaccination,
)
from coronavirus.visualization import plot_data, plot_data_by_country
from coronavirus.tswindow import WindowGenerator
from coronavirus.tsdata import split_for_training
from coronavirus.models import MultipleStepRNN
from sklearn.decomposition import PCA
from sklearn import preprocessing


(
    confirmed_by_country,
    deaths_by_country,
    recovered_by_country,
    recovered_or_passed_by_country,
    infected_by_country,
) = extract_process_data()

COUNTRY_NAMES = ["United Kingdom", "France", "Germany", "Italy"]
# COUNTRY_NAMES = ["France"]
plot_data_by_country(
    [
        confirmed_by_country.iloc[:, -112:],
        deaths_by_country.iloc[:, -112:],
    ],
    COUNTRY_NAMES,
    rolling_mean_window=7,
    diff=0,
)

# |         | UK         | FRANCE     |
# | Pop     | 67 886 004 | 67 848 156 |
# | Density | 270.7/km2  | 107,2/km2  |

df_vac_uk = pd.read_csv("UK-data_2021-Mar-05.csv")
df_vac_uk["date"] = pd.to_datetime(df_vac_uk["date"])
df_vac_uk = df_vac_uk.sort_values("date")
df_vac_uk = df_vac_uk.set_index("date")
df_vac_uk["cumPeopleVaccinatedFirstDoseByPublishDate"].plot(
    figsize=(14, 10), x_compat=True
)
plt.grid(True)
plt.legend(loc="upper left")

train_df, validation_df, test_df = split_for_training(
    confirmed_by_country.loc["France", :].reset_index(drop=True).to_frame()
)

OUT_STEPS = 20
window_generator = WindowGenerator(
    input_width=7,
    label_width=OUT_STEPS,
    shift=OUT_STEPS,
    train_df=train_df,
    validation_df=validation_df,
    test_df=test_df,
    label_columns=COUNTRY_NAMES,
)

# MultipleStepRNN
multiple_step_rnn = MultipleStepRNN(
    output_steps=OUT_STEPS,
    label_index=window_generator.column_indices["France"],
)

multiple_step_rnn.compile_and_fit(window_generator)

# Visualization
window_generator.plot(
    *next(iter(window_generator.validation)),
    model=multiple_step_rnn,
    label_column="France",
    max_subplots=3,
)