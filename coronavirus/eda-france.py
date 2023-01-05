import pandas as pd
from matplotlib import pyplot as plt

from coronavirus.extraction import (
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


def scale(df):
    min_max = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(
        min_max.fit_transform(df),
        columns=df.columns,
    )
    df_scaled.index = df.index
    return df_scaled


def pca_2_components(df, index=None):
    pca = PCA(n_components=2)
    pca.fit(df.to_numpy())
    df_reduced = pca.transform(df.to_numpy())
    index = index if index is not None else df.index

    plt.scatter(
        x=index,
        y=df_reduced[:, 0],
        label="1st",
    )
    plt.scatter(
        x=index,
        y=df_reduced[:, 1],
        label="2nd",
    )
    plt.legend(loc="best")
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    return df_reduced


df_vue_ensemble = data_gouv_vue_ensemble()
df_taux_incidence, df_65, df_0 = data_gouv_taux_incidence()
df_hosp, df_rea, df_dc = data_gouv_hospital()
df_region, df_national = data_gouv_vaccination()

# Vue d'ensemble
scale(df_vue_ensemble).plot(figsize=(14, 10), x_compat=True)
plt.grid(True)
plt.legend(loc="upper left")

df_vue_ensemble["total_deces_hopital"].diff().dropna().rolling(
    window=7
).mean().dropna().plot(figsize=(14, 10), x_compat=True)
plt.grid(True)
plt.legend(loc="upper left")

df_vue_ensemble["total_deces_ehpad"].diff().dropna().rolling(
    window=7
).mean().dropna().plot(figsize=(14, 10), x_compat=True)
plt.grid(True)
plt.legend(loc="upper left")

df_vue_ensemble["nouveaux_patients_reanimation"].rolling(window=7).mean().dropna().plot(
    figsize=(14, 10), x_compat=True
)
plt.grid(True)
plt.legend(loc="upper left")

# taux incidence + de 65
df_65.plot(figsize=(14, 10), x_compat=True)
plt.grid(True)
plt.title("Taux d'incidence total +65")
plt.legend(loc="upper left")


df_65.diff().dropna().rolling(window=7).mean().dropna().plot(
    figsize=(14, 10), x_compat=True
)
plt.grid(True)
plt.legend(loc="upper left")


# taux incidence total
df_0.plot(figsize=(14, 10), x_compat=True)
plt.grid(True)
plt.title("Taux d'incidence total")
plt.legend(loc="upper left")

df_0.diff().dropna().rolling(window=7).mean().dropna().plot(
    figsize=(14, 10), x_compat=True
)
plt.grid(True)
plt.legend(loc="upper left")


df_65["Lille"].diff().dropna().rolling(window=7).mean().dropna().plot(
    figsize=(14, 10), x_compat=True
)
plt.grid(True)
plt.title("Taux d'incidence total +65 Lille")
plt.legend(loc="upper left")

# PCA
df_0_reduced = pca_2_components(df_0[-30:])
df_65_reduced = pca_2_components(df_65)

# Hospitalisation
weeks = 20

df_hosp[-90:].plot(figsize=(14, 10), x_compat=True)
plt.grid(True)
plt.title("Hospitalisations")
plt.legend(loc="upper left")

df_hosp[-(7 * weeks) :].diff().dropna().rolling(window=14).mean().dropna().plot(
    figsize=(14, 10), x_compat=True, legend=None
)
plt.title("Diff lissé Hospitalisations par département")
plt.grid(True)

# PCA
days = 7
df_hosp_reduced = pca_2_components(df_hosp[-days:])

# Réanimations
df_rea[-(7 * weeks) :].plot(figsize=(14, 10), x_compat=True, legend=None)
plt.title("Réanimations par département")
plt.grid(True)

df_rea[-(7 * weeks) :].diff().dropna().rolling(window=14).mean().dropna().plot(
    figsize=(14, 10), x_compat=True, legend=None
)
plt.title("Diff lissé réanimations par département")
plt.grid(True)

## PCA
days = 21
df_rea_reduced = pca_2_components(df_rea[-days:])

## Scale
df_rea_scaled = scale(df_rea)
df_rea_scaled.plot(figsize=(14, 10), x_compat=True, legend=None)
plt.grid(True)

## PCA scaled
df_rea_scaled_reduced = pca_2_components(
    df_rea_scaled[-days:],
    index=df_rea[-days:].index,
)

# Vaccinations
df_region["n_cum_dose1"].plot(figsize=(14, 10), x_compat=True)
plt.title("Vaccinations par région")
plt.grid(True)

df_national["n_cum_dose1"].plot(figsize=(14, 10), x_compat=True)
plt.title("Vaccinations nationale")
plt.grid(True)
