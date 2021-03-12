from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def date_from_string(date_string):
    return datetime.strptime(date_string, "%m/%d/%y")


def delta(date_1, date_2):
    delta = date_from_string(date_1) - date_from_string(date_2)
    return delta.days


def plot_data(df):
    x = np.arange(df.shape[0])
    y = df.to_numpy()

    f, ax = plt.subplots(figsize=(14, 10))
    plt.plot(x, y, ".")
    plt.xlabel("Days")
    plt.ylabel("Confirmed", rotation="vertical")
    plt.grid(color="gray", ls="-.", lw=0.25)

    # Time axis
    date_origin = date_from_string(df.index[0])
    date_end = date_from_string(df.index[-1])
    date_range = [
        date_origin + timedelta(days=x)
        for x in range(0, (date_end - date_origin).days + 1)
    ]

    ax.set_xticks(np.arange(df.shape[0]))
    ax.set_xticklabels([datetime.strftime(x, "%d-%m-%Y") for x in date_range])

    plt.show()


def plot_data_by_country(dfs, countries, rolling_mean_window=0, diff=0):
    max_n = len(dfs)

    fig = plt.figure(figsize=(14, 10))
    for i, df in enumerate(dfs):

        ax = plt.subplot(max_n, 1, i + 1)
        ax.ticklabel_format(useOffset=False)

        time_index = df.index.get_indexer(["Time"])
        if rolling_mean_window == 0:
            x = dates.date2num(df.iloc[time_index, diff:])[0]
        else:
            x = dates.date2num(df.iloc[time_index, rolling_mean_window - 1 + diff :])[0]

        for country in countries:
            country_index = df.index.get_indexer([country])

            if rolling_mean_window == 0:
                df_country = df.iloc[country_index, :].transpose()
            else:
                df_country = (
                    df.iloc[country_index, :]
                    .rolling(window=rolling_mean_window, axis=1)
                    .mean()
                    .transpose()
                )

            if diff > 0:
                for _ in range(diff):
                    df_country = df_country.diff()

            df_country = df_country.dropna().to_numpy()

            ax.plot_date(
                x,
                df_country,
                fmt=".",
                label=country,
            )

        ax.xaxis.set_ticks(
            np.arange(min(x), max(x) + 1, 1 if len(x) < 10 else int(len(x) / 10))
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

        plt.grid(True)
        plt.legend()

    plt.show()
