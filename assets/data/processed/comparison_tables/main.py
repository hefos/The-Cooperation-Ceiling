import numpy as np
import pathlib
import argparse
import dask.dataframe as dd

here = pathlib.Path(__file__).resolve()
data_path = here.parents[2]

parser = argparse.ArgumentParser(description="Processes to compare")

parser.add_argument("--p1", type=str, required=True, help="A string value for dynamic one")
parser.add_argument("--p2", type=str, required=True, help="A string value for dynamic two")

args = parser.parse_args()

p1_string = args.p1
p2_string = args.p2

dataframe_dict = {
    "Fermi":[
    data_path / "raw/fermi_mu_eq_0.1/main.csv",
    data_path / "raw/fermi_mu_eq_0.001/main.csv",
    data_path / "raw/fermi_mu_eq_0.005/main.csv",
    data_path / "raw/fermi_mu_eq_0.05/main.csv"
    ],
    "aspiration":[
    data_path / "raw/aspiration_mu_eq_0.1/main.csv",
    data_path / "raw/aspiration_mu_eq_0.001/main.csv",
    data_path / "raw/aspiration_mu_eq_0.005/main.csv",
    data_path / "raw/aspiration_mu_eq_0.05/main.csv"
    ],
    "imispection":[
    data_path / "raw/imispection_mu_eq_0.1/main.csv",
    data_path / "raw/imispection_mu_eq_0.001/main.csv",
    data_path / "raw/imispection_mu_eq_0.005/main.csv",
    data_path / "raw/imispection_mu_eq_0.05/main.csv"
    ],
    "introspection":[
    data_path / "raw/introspection_mu_eq_0.1/main.csv",
    data_path / "raw/introspection_mu_eq_0.001/main.csv",
    data_path / "raw/introspection_mu_eq_0.005/main.csv",
    data_path / "raw/introspection_mu_eq_0.05/main.csv"
    ],
    "Moran":[
    data_path / "raw/moran_mu_eq_0.1/main.csv",
    data_path / "raw/moran_mu_eq_0.001/main.csv",
    data_path / "raw/moran_mu_eq_0.005/main.csv",
    data_path / "raw/moran_mu_eq_0.05/main.csv"
]
}

significant_parameter_dict = {
    "aspiration":["N","M","r","beta","aspiration","mu"],
    "Fermi":["N","M","r","beta","mu"],
    "introspection":["N","M","r","beta","mu"],
    "imispection":["N","M","r","beta","epsilon","mu"],
    "Moran":["N","M","r","epsilon","mu"],
}

merge_conditions = [x for x in significant_parameter_dict[p1_string] if x in significant_parameter_dict[p2_string]]

df_1 = dd.read_csv(dataframe_dict[p1_string])
df_2 = dd.read_csv(dataframe_dict[p2_string])

df = df_1.merge(
    df_2,
    on=merge_conditions,
    how="inner",
    suffixes=(f"_in_{p1_string}", f"_in_{p2_string}")
)

df = df.drop(columns=[ f"alpha_i_in_{p1_string}", f"i_in_{p1_string}", f"alpha_i_in_{p2_string}", f"i_in_{p2_string}"], errors="ignore")

conditions = [
    df[f"p_C_in_{p1_string}"] > df[f"p_C_in_{p2_string}"],
    df[f"p_C_in_{p1_string}"] < df[f"p_C_in_{p2_string}"],
    df[f"p_C_in_{p1_string}"] == df[f"p_C_in_{p2_string}"],
]

choices = [p1_string, p2_string, "draw"]

df=df.compute()

df["winner"] = np.select(conditions, choices, default="draw")
df["winner_margin"] = np.abs(
    df[f"p_C_in_{p1_string}"] - df[f"p_C_in_{p2_string}"]
)


folder = pathlib.Path(here.parent / f"{p1_string}_vs_{p2_string}")
folder.mkdir(exist_ok=True)

df.to_csv(here.parent / f"{p1_string}_vs_{p2_string}/main.csv")
