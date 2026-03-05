import pandas as pd
import numpy as np
from pathlib import Path
import fractions


here = Path(__file__).resolve()
asset_path = here.parents[3]

df = pd.read_csv(asset_path / "data/processed/comparison_tables/table/main.csv")
df = df[df["population"] == "linear"]

table_columns = df["process_numerator"].unique()

table = {}

for i, process_1 in enumerate(table_columns):
    data = []
    for j, process_2 in enumerate(table_columns):
        if i == j:
            data.append("N/A")
            continue

        df_row = df[df["process_numerator"] == process_1]
        df_row = df_row[df_row["process_other"] == process_2]
        data.append(
            f"\\frac{{ {df_row["wins"].iloc[0]} }}{{ {df_row["total_compared"].iloc[0]} }}"
        )
    table[process_1] = data

with open(
    here.parent / "main.tex",
    "w",
) as f:
    f.write(r"\begin{tabular}")
    f.write("{c|c|c|c|c}\n")
    header = " & " + " & ".join(table_columns) + "\\\\"
    f.write(header)
    f.write("\n")
    for process, result in table.items():
        f.write(process)
        f.write("&")
        for entry in result:
            print(entry)
            f.write(entry)
            f.write("&")
        f.write("\\\\")
        f.write("\n")
    f.write("\end{tabular}")
