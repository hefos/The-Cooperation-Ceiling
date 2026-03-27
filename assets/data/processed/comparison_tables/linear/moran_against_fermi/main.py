import pandas as pd
import numpy as np
from pathlib import Path
import sys


here = Path(__file__).resolve()
root_path = here.parents[6]

sys.path.append(str(root_path))

df_moran = pd.read_csv(root_path / "assets/data/raw/linear_population_moran/main.csv")
df_fermi = pd.read_csv(root_path / "assets/data/raw/linear_population_fermi/main.csv")

df_moran = df_moran.drop(columns=["process", "stochastic"])
if "first_alpha" in df_moran.columns:
    df_moran.rename(columns={"first_alpha": "mutant_alpha"}, inplace=True)
data_group = df_moran.groupby("UID")["alpha_i"]
df_moran["alpha_range"] = data_group.transform("max") - data_group.transform("min")
df_moran["std_alpha"] = data_group.transform("std")
df_moran = df_moran.groupby("UID").first()


df_fermi = df_fermi.drop(columns=["process", "stochastic"])
if "first_alpha" in df_fermi.columns:
    df_fermi.rename(columns={"first_alpha": "mutant_alpha"}, inplace=True)
data_group = df_fermi.groupby("UID")["alpha_i"]
df_fermi["alpha_range"] = data_group.transform("max") - data_group.transform("min")
df_fermi["std_alpha"] = data_group.transform("std")
df_fermi = df_fermi.groupby("UID").first()


df_compare = df_moran.merge(
    df_fermi,
    on=["mutant_alpha", "N", "r", "population", "alpha_range", "std_alpha"],
    how="inner",
    suffixes=("_in_moran", "_in_fermi"),
)

df_compare = df_compare.drop(
    columns=["alpha_i_in_moran", "i_in_moran", "alpha_i_in_fermi", "i_in_fermi"]
)

conditions = [
    df_compare["p_C_in_moran"] > df_compare["p_C_in_fermi"],
    df_compare["p_C_in_moran"] < df_compare["p_C_in_fermi"],
    df_compare["p_C_in_moran"] == df_compare["p_C_in_fermi"],
]

choices = ["moran", "fermi", "draw"]

df_compare["winner"] = np.select(conditions, choices, default="draw")
df_compare["winner_margin"] = np.abs(
    df_compare["p_C_in_moran"] - df_compare["p_C_in_fermi"]
)

df_compare.to_csv(here.parent / "main.csv")
