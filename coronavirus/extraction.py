import os
import pandas as pd

# https://github.com/CSSEGISandData/COVID-19.git

REPOSITORY = "https://raw.githubusercontent.com/CSSEGISandData"
MAIN_FOLDER = "COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"

CONFIRMED_FILE = "time_series_covid19_confirmed_global.csv"
DEATHS_FILE = "time_series_covid19_deaths_global.csv"
RECOVERED_FILE = "time_series_covid19_recovered_global.csv"

CONFIRMED_PATH = os.path.join(REPOSITORY, MAIN_FOLDER, CONFIRMED_FILE)
DEATHS_PATH = os.path.join(REPOSITORY, MAIN_FOLDER, DEATHS_FILE)
RECOVERED_PATH = os.path.join(REPOSITORY, MAIN_FOLDER, RECOVERED_FILE)


def group_data_by_country(df):
    df = df.drop(columns=["Lat", "Long"])
    df_bycountry = df.groupby("Country/Region").sum()

    # summing for all country
    df_bycountry.loc["Total"] = df_bycountry.sum(axis=0)

    return df_bycountry


def get_data_normalized(df):
    # dividing by the sum
    maximums = df.iloc[:, -1]
    df_normalized = df.div(maximums.to_numpy(), axis=0)

    return df_normalized


def get_data_for_sir(df_death, df_recovered, df_confirmed):
    df_recovered_or_passed = df_recovered + df_death
    df_infected = df_confirmed - df_recovered_or_passed

    return df_recovered_or_passed, df_infected


def extract_process_data():
    df_confirmed = pd.read_csv(CONFIRMED_PATH)
    df_deaths = pd.read_csv(DEATHS_PATH)
    df_recovered = pd.read_csv(RECOVERED_PATH)

    df_confirmed_by_country = group_data_by_country(df_confirmed)
    df_deaths_by_country = group_data_by_country(df_deaths)
    df_recovered_by_country = group_data_by_country(df_recovered)

    df_recovered_or_passed_by_country, df_infected_by_country = get_data_for_sir(
        df_deaths_by_country, df_recovered_by_country, df_confirmed_by_country
    )

    return (
        add_datetime(df_confirmed_by_country),
        add_datetime(df_deaths_by_country),
        add_datetime(df_recovered_by_country),
        add_datetime(df_recovered_or_passed_by_country),
        add_datetime(df_infected_by_country),
    )


def add_datetime(df):
    df.loc["Time"] = pd.period_range(df.columns[0], df.columns[-1], freq="D")
    return df