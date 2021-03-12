import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df_male = pd.read_csv("doc/pyramid-by-dep-male.csv")
df_female = pd.read_csv("doc/pyramid-by-dep-female.csv")

male = (
    df_male[df_male["Region"] == "Ain"].drop(columns=["Code", "Region"]).transpose()
    * -1
)
male.columns = ["Male"]

female = (
    df_female[df_female["Region"] == "Ain"].drop(columns=["Code", "Region"]).transpose()
)
female.columns = ["Female"]

df_dep = pd.concat([male, female], axis=1)
df_dep = df_dep.reset_index().rename(columns={"index": "Age"})

age_order = df_dep["Age"].to_list()[::-1]

plt.figure(figsize=(14, 10))
bar_plot = sns.barplot(
    x="Male",
    y="Age",
    data=df_dep,
    order=age_order,
    color="tab:blue",
    lw=0,
)
bar_plot = sns.barplot(
    x="Female",
    y="Age",
    data=df_dep,
    order=age_order,
    color="tab:red",
    lw=0,
)

bar_plot.set_xticklabels(np.abs(bar_plot.get_xticks()).astype(int))
bar_plot.set(xlabel="Population", ylabel="Age-Group", title="Population Pyramid")
