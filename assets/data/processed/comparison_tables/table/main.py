import pandas as pd
from pathlib import Path


here = Path(__file__).resolve()
comparison_path = here.parents[1]

data_paths = [
    [
        comparison_path / "binomial/imispection_against_fermi/main.csv",
        "introspective imitation",
        "fermi",
    ],
    [
        comparison_path / "binomial/moran_against_imispection/main.csv",
        "moran",
        "introspective imitation",
    ],
    [comparison_path / "binomial/moran_against_fermi/main.csv", "moran", "fermi"],
    [
        comparison_path / "binomial/introspection_against_fermi/main.csv",
        "introspection",
        "fermi",
    ],
    [
        comparison_path / "binomial/introspection_against_imispection/main.csv",
        "introspection",
        "introspective imitation",
    ],
    [
        comparison_path / "binomial/introspection_against_moran/main.csv",
        "introspection",
        "moran",
    ],
    [comparison_path / "linear/moran_against_fermi/main.csv", "moran", "fermi"],
    [
        comparison_path / "linear/moran_against_imispection/main.csv",
        "moran",
        "introspective imitation",
    ],
    [
        comparison_path / "linear/imispection_against_fermi/main.csv",
        "introspective imitation",
        "fermi",
    ],
    [
        comparison_path / "linear/introspection_against_fermi/main.csv",
        "introspection",
        "fermi",
    ],
    [
        comparison_path / "linear/introspection_against_imispection/main.csv",
        "introspection",
        "introspective imitation",
    ],
    [
        comparison_path / "linear/introspection_against_moran/main.csv",
        "introspection",
        "moran",
    ],
]


df = pd.DataFrame(
    columns=[
        "process_numerator",
        "process_other",
        "population",
        "wins",
        "total_compared",
        "win_ratio",
    ]
)
df.to_csv(here.parent / "main.csv", index=False)

data = []
for df_path, process_1, process_2 in data_paths:
    print(process_1, process_2)
    df = pd.read_csv(df_path)
    print("read")
    total_compared = len(df)
    wins = (df["winner"] == process_1).sum()
    win_ratio = wins / total_compared
    population = df["population"].iloc[0]
    row = [process_1, process_2, population, wins, total_compared, win_ratio]
    data.append(row)

    wins_2 = (df["winner"] == process_2).sum()
    win_ratio_2 = wins_2 / total_compared
    row = [process_2, process_1, population, wins_2, total_compared, win_ratio_2]
    data.append(row)

df = pd.DataFrame(data)
df.to_csv(here.parent / "main.csv", mode="a", header=False, index=False)
