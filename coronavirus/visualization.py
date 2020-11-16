import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import pandas as pd


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


def plot_data_by_country(df, country):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.ticklabel_format(useOffset=False)
    ts = pd.Series(
        df.loc[country, :].to_numpy(),
        index=df.loc["Time", :],
    )
    ts.plot()
