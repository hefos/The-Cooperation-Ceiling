import pandas as pd
import numpy as np
from pathlib import Path
import sys


here = Path(__file__).resolve()
root_path = here.parents[6]

sys.path.append(str(root_path))

df_imispection = pd.read_csv(
    root_path / "assets/data/raw/binomial_population_imispection/main.csv"
)
df_introspection = pd.read_csv(
    root_path / "assets/data/raw/binomial_population_introspection/main.csv"
)

df_imispection = df_imispection.drop(columns=["process", "stochastic"])
if "first_alpha" in df_imispection.columns:
    df_imispection.rename(columns={"first_alpha": "mutant_alpha"}, inplace=True)
if "population_type" in df_imispection.columns:
    df_imispection.rename(columns={"population_type": "population"}, inplace=True)
data_group = df_imispection.groupby("UID")["alpha_i"]
df_imispection["alpha_range"] = data_group.transform("max") - data_group.transform(
    "min"
)
df_imispection["std_alpha"] = data_group.transform("std")
df_imispection = df_imispection.groupby("UID").first()


df_introspection = df_introspection.drop(columns=["process", "stochastic", "i_C"])
if "first_alpha" in df_introspection.columns:
    df_introspection.rename(columns={"first_alpha": "mutant_alpha"}, inplace=True)
data_group = df_introspection.groupby("UID")["alpha_i"]
df_introspection["alpha_range"] = data_group.transform("max") - data_group.transform(
    "min"
)
df_introspection["std_alpha"] = data_group.transform("std")
df_introspection = df_introspection.groupby("UID").first()


df_compare = df_imispection.merge(
    df_introspection,
    on=["N", "r", "population", "beta", "alpha_range", "std_alpha"],
    how="inner",
    suffixes=("_in_imispection", "_in_introspection"),
)

df_compare = df_compare.drop(
    columns=[
        "alpha_i_in_imispection",
        "i_in_imispection",
        "alpha_i_in_introspection",
        "i_in_introspection",
    ]
)

conditions = [
    df_compare["p_C_in_imispection"] > df_compare["p_C_in_introspection"],
    df_compare["p_C_in_imispection"] < df_compare["p_C_in_introspection"],
    df_compare["p_C_in_imispection"] == df_compare["p_C_in_introspection"],
]

choices = ["introspective imitation", "introspection", "draw"]

df_compare["winner"] = np.select(conditions, choices, default="draw")
df_compare["winner_margin"] = np.abs(
    df_compare["p_C_in_imispection"] - df_compare["p_C_in_introspection"]
)

df_compare.to_csv(here.parent / "main.csv")
