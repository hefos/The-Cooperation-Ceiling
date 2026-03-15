import pandas as pd
import pathlib

file_path = pathlib.Path(__file__)
from stet.backends import get_backend

df = pd.read_csv(file_path.parent / "main.csv")
backend = get_backend(pathlib.Path(file_path.parent / "_stet_store.sqlite"))
for uid, experiment_frame in df.groupby("UID"):
    M_in_experiment = experiment_frame["alpha_i"].sum()
    for _, row in experiment_frame.iterrows():
        backend.record(
            {
                "M": M_in_experiment,
                "r": row["r"],
                "choice_intensity": row["beta"],
                "first_contribution": row["mutant_alpha"],
                "N": row["N"],
                "i": row["i"],
                "selection_intensity": row["epsilon"],
            }
        )
