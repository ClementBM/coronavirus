import pandas as pd
from matplotlib import pyplot as plt

from coronavirus.extraction import (
    data_gouv_vue_ensemble,
    data_gouv_taux_incidence,
    data_gouv_hospital,
    data_gouv_vaccination,
    extract_process_data,
    data_gouv_vue_ensemble,
    data_gouv_taux_incidence,
    data_gouv_hospital,
    data_gouv_vaccination,
)
from coronavirus.visualization import plot_data, plot_data_by_country

from sklearn import preprocessing


def scale(df):
    min_max = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(
        min_max.fit_transform(df),
        columns=df.columns,
    )
    df_scaled.index = df.index
    return df_scaled


# https://coronavirus.data.gov.uk/details/vaccinations
# |         | UK         | FRANCE     |
# | Pop     | 67 886 004 | 67 848 156 |
# | Density | 270.7/km2  | 107,2/km2  |

COUNTRY_NAMES = ["United Kingdom", "France"]

df_region, df_national = data_gouv_vaccination()


df_vac_uk = pd.read_csv("UK-data_2021-Mar-05.csv")
df_vac_uk["date"] = pd.to_datetime(df_vac_uk["date"])
df_vac_uk = df_vac_uk.sort_values("date")
df_vac_uk = df_vac_uk.set_index("date")

# Merge vaccines data
df_vac_uk_1 = df_vac_uk["cumPeopleVaccinatedFirstDoseByPublishDate"]
df_vac_fr_1 = df_national["n_cum_dose1"]

df_vac_uk_1 = df_vac_uk_1.reset_index()
df_vac_fr_1 = df_vac_fr_1.reset_index()

df_uk_fr = pd.merge(df_vac_uk_1, df_vac_fr_1, how="outer", on="date")
df_uk_fr = df_uk_fr.sort_values("date")

df_uk_fr = df_uk_fr.rename(
    columns={
        "cumPeopleVaccinatedFirstDoseByPublishDate": "uk",
        "n_cum_dose1": "fr",
    }
)

df_uk_fr.index = pd.to_datetime(df_uk_fr["date"])
df_uk_fr = df_uk_fr.drop(columns=["date"])

df_uk_fr.plot(figsize=(14, 10), x_compat=True)
plt.title("Vaccinations by country")
plt.grid(True)

(
    confirmed_by_country,
    deaths_by_country,
    recovered_by_country,
    recovered_or_passed_by_country,
    infected_by_country,
) = extract_process_data()


def prepare_by_country(df, country, name):
    df_prepared = df.loc[country].transpose()
    df_prepared.index = pd.to_datetime(df_prepared.index)
    df_prepared = df_prepared.reset_index()
    df_prepared = df_prepared.rename(
        columns={"index": "date", country: f"{country} {name}"}
    )
    return df_prepared


uk_confirmed = prepare_by_country(confirmed_by_country, "United Kingdom", "confirmed")
uk_deaths = prepare_by_country(deaths_by_country, "United Kingdom", "death")
fr_confirmed = prepare_by_country(confirmed_by_country, "France", "confirmed")
fr_deaths = prepare_by_country(deaths_by_country, "France", "death")

uk_confirmed_death = pd.merge(uk_confirmed, uk_deaths, on="date")
fr_confirmed_death = pd.merge(fr_confirmed, fr_deaths, on="date")

fr_uk_deaths = pd.merge(uk_deaths, fr_deaths, on="date")
uk_fr_death_diff = (
    fr_uk_deaths.set_index("date").diff().dropna().rolling(window=7).mean().dropna()
)

uk_fr_confirmed_death = pd.merge(uk_confirmed_death, fr_confirmed_death, on="date")

uk_fr_confirmed_death_diff = (
    uk_fr_confirmed_death.set_index("date")
    .diff()
    .dropna()
    .rolling(window=7)
    .mean()
    .dropna()
)
uk_fr_confirmed_death_diff.iloc[
    -68:,
].plot(figsize=(14, 10), x_compat=True)
plt.title("UK / FR")
plt.grid(True)

uk_fr_death_diff.iloc[
    -68:,
].plot(figsize=(14, 10), x_compat=True)
plt.title("UK / FR")
plt.grid(True)

uk_fr_all = pd.merge(df_uk_fr, uk_fr_confirmed_death_diff, how="left", on="date")

uk_fr_all.plot(figsize=(14, 10), x_compat=True)
plt.title("UK / FR")
plt.grid(True)

scale(uk_fr_all).plot(figsize=(14, 10), x_compat=True)
plt.title("UK / FR")
plt.grid(True)

# Vaccinations
df_region["n_cum_dose1"].plot(figsize=(14, 10), x_compat=True)
plt.title("Vaccinations par région")
plt.grid(True)

df_national["n_cum_dose1"].plot(figsize=(14, 10), x_compat=True)
plt.title("Vaccinations nationale")
plt.grid(True)

plot_data_by_country(
    [
        confirmed_by_country.iloc[:, -77:],
        deaths_by_country.iloc[:, -77:],
    ],
    ["United Kingdom", "France"],
    rolling_mean_window=7,
    diff=1,
)

df_vac_uk["cumPeopleVaccinatedFirstDoseByPublishDate"].plot(
    figsize=(14, 10), x_compat=True
)
plt.grid(True)
plt.legend(loc="upper left")
