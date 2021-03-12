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


def data_gouv_vue_ensemble():
    """
    Données relatives à l’épidémie de COVID-19 en France : vue d’ensemble
    https://www.data.gouv.fr/fr/datasets/donnees-relatives-a-lepidemie-de-covid-19-en-france-vue-densemble/#_

    Columns:
    date
    total_cas_confirmes
    total_deces_hopital
    total_deces_ehpad
    total_cas_confirmes_ehpad
    total_cas_possibles_ehpad
    patients_reanimation
    patients_hospitalises
    total_patients_gueris
    nouveaux_patients_hospitalises
    nouveaux_patients_reanimation
    """
    url_stable = (
        "https://www.data.gouv.fr/fr/datasets/r/d3a98a30-893f-47f7-96c5-2f4bcaaa0d71"
    )
    df = pd.read_csv(url_stable)

    df.index = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"])
    df = df.sort_index()

    return df


def data_gouv_taux_incidence():
    """
    Indicateurs de l’activité épidémique:
    taux d'incidence de l'épidémie de COVID-19 par métropole
    https://www.data.gouv.fr/fr/datasets/indicateurs-de-lactivite-epidemique-taux-dincidence-de-lepidemie-de-covid-19-par-metropole/

    Columns:
    epci2020: Code EPCI
    semaine_glissante: Semaine glissante
    clage_65:
        0 si taux d'incidence toute classe d'âge
        65 si taux d'incidence pour les personnes âgées de plus de 65 ans
    ti: Nombre de nouveaux cas positifs pour 100 000 habitants sur 7 jours glissants
    """
    url_stable = (
        "https://www.data.gouv.fr/fr/datasets/r/61533034-0f2f-4b16-9a6d-28ffabb33a02"
    )
    df_main = pd.read_csv(url_stable)
    df_main = df_main.rename(columns={"epci2020": "EPCI"})

    # Correspondences entre le code EPCI et le nom des métropoles
    url_epci = "doc/metropole-epci.csv"
    df_epci = pd.read_csv(url_epci, sep=";")

    df = pd.merge(df_main, df_epci, how="left", on="EPCI")

    df_65 = df.loc[df["clage_65"] == 65]
    df_65["semaine_glissante"] = [i[11:] for i in df_65["semaine_glissante"]]
    df_65.index = pd.to_datetime(df_65["semaine_glissante"])
    df_65 = df_65.drop(columns=["semaine_glissante", "clage_65", "EPCI"])
    df_65 = df_65.pivot(columns="Metropole", values="ti")

    df_0 = df.loc[df["clage_65"] == 0]
    df_0["semaine_glissante"] = [i[11:] for i in df_0["semaine_glissante"]]
    df_0.index = pd.to_datetime(df_0["semaine_glissante"])
    df_0 = df_0.drop(columns=["semaine_glissante", "clage_65", "EPCI"])
    df_0 = df_0.pivot(columns="Metropole", values="ti")

    return df, df_65, df_0


def data_gouv_hospital():
    """
    Les données relatives à les nouvelles admissions en réanimation par région :
    nombre de nouveaux patients admis en réanimation dans les 24 dernières heures.

    Columns:
    sexe:
        0: femmes + hommes
        1: hommes
        2: femmes
    dep: Département
    hosp: Nombre de personnes actuellement hospitalisées
    rea: Nombre de personnes actuellement en réanimation ou soins intensifs
    rad: Nombre cumulé de personnes retournées à domicile
    dc: Nombre cumulé de personnes décédées à l'hôpital
    """
    url_stable = (
        "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"
    )

    df_main = pd.read_csv(url_stable, sep=";")

    df_main.index = pd.to_datetime(df_main["jour"])
    df_main = df_main.loc[df_main["sexe"] == 0]

    df_hosp = df_main.drop(columns=["sexe", "jour", "rea", "rad", "dc"])
    df_hosp = df_hosp.pivot(columns="dep", values="hosp")

    df_rea = df_main.drop(columns=["sexe", "jour", "hosp", "rad", "dc"])
    df_rea = df_rea.pivot(columns="dep", values="rea")

    df_dc = df_main.drop(columns=["sexe", "jour", "rea", "rad", "hosp"])
    df_dc = df_dc.pivot(columns="dep", values="dc")

    return df_hosp, df_rea, df_dc


def data_gouv_vaccination():
    """
    https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1/#_

    Columns:
    reg
    sexe:
        0 : hommes + femmes + Non renseigné
        1 : homme
        2 : femme
    n_dose1
    n_dose2
    n_cum_dose1
    n_cum_dose2
    """

    url_regional = (
        "https://www.data.gouv.fr/fr/datasets/r/96db2c1a-8c0c-413c-9a07-f6f62f3d4daf"
    )
    url_national = (
        "https://www.data.gouv.fr/fr/datasets/r/349ca785-cf12-4f4d-9a0a-846d53dce996"
    )

    df_region = pd.read_csv(url_regional, sep=";")
    df_national = pd.read_csv(url_national, sep=";")

    df_region = df_region.rename(columns={"jour": "date"})
    df_national = df_national.rename(columns={"jour": "date"})

    df_region.index = pd.to_datetime(df_region["date"])
    df_national.index = pd.to_datetime(df_national["date"])

    df_region = df_region.loc[df_region["sexe"] == 0]
    df_national = df_national.loc[df_national["sexe"] == 0]

    df_region = df_region.drop(columns=["date", "sexe"])
    df_national = df_national.drop(columns=["date", "sexe"])

    return df_region, df_national