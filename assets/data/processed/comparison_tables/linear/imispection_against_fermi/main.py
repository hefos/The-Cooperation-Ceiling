import pandas as pd

print("hello")
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402


here = Path(__file__).resolve()
root_path = here.parents[6]

sys.path.append(str(root_path))

df_imispection = pd.read_csv(
    root_path / "assets/data/raw/linear_population_imispection/main.csv"
)
df_fermi = pd.read_csv(root_path / "assets/data/raw/linear_population_fermi/main.csv")

df_imispection = df_imispection.drop(columns=["process", "stochastic"])
if "first_alpha" in df_imispection.columns:
    df_imispection.rename(columns={"first_alpha": "mutant_alpha"}, inplace=True)
data_group = df_imispection.groupby("UID")["alpha_i"]
df_imispection["alpha_range"] = data_group.transform("max") - data_group.transform(
    "min"
)
df_imispection["std_alpha"] = data_group.transform("std")
df_imispection = df_imispection.groupby("UID").first()


df_fermi = df_fermi.drop(columns=["process", "stochastic"])
if "first_alpha" in df_fermi.columns:
    df_fermi.rename(columns={"first_alpha": "mutant_alpha"}, inplace=True)
data_group = df_fermi.groupby("UID")["alpha_i"]
df_fermi["alpha_range"] = data_group.transform("max") - data_group.transform("min")
df_fermi["std_alpha"] = data_group.transform("std")
df_fermi = df_fermi.groupby("UID").first()


df_compare = df_imispection.merge(
    df_fermi,
    on=["mutant_alpha", "N", "r", "population", "beta", "alpha_range", "std_alpha"],
    how="inner",
    suffixes=("_in_imispection", "_in_fermi"),
)

df_compare = df_compare.drop(
    columns=[
        "alpha_i_in_imispection",
        "i_in_imispection",
        "alpha_i_in_fermi",
        "i_in_fermi",
    ]
)

conditions = [
    df_compare["p_C_in_imispection"] > df_compare["p_C_in_fermi"],
    df_compare["p_C_in_imispection"] < df_compare["p_C_in_fermi"],
    df_compare["p_C_in_imispection"] == df_compare["p_C_in_fermi"],
]

choices = ["introspective imitation", "fermi", "draw"]

df_compare["winner"] = np.select(conditions, choices, default="draw")
df_compare["winner_margin"] = np.abs(
    df_compare["p_C_in_imispection"] - df_compare["p_C_in_fermi"]
)

df_compare.to_csv(here.parent / "main.csv")
