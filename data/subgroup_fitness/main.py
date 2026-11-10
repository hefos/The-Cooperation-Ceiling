import pandas as pd
import pathlib
import numpy as np
import itertools

file_path = pathlib.Path(__file__).resolve()

d = 3
r_high = 6
r_low = 3.1
alpha = 2
state = np.array([0,1,1,1,1,0,0,0])
columns = [
            "number of cooperators",
            "r",
            "type",
            "payoff"
        ]
df = pd.DataFrame(
        columns=columns
    )

df.to_csv(file_path.parent / "main.csv", index=False)


def count_pairs(state, index, number_of_cooperators, d):
    remaining = np.delete(state, index)
    return sum(
        sum(pair) == number_of_cooperators-state[index]
        for pair in itertools.combinations(remaining, d-1)
    )

def payoff_for_number(r, alpha, number_of_cooperators, d, type):
    public_good = r*alpha*number_of_cooperators/d
    cost = alpha * type
    return public_good - cost

def get_total_payoff_from_makeup(r, alpha, d, index, state, number_of_cooperators):
    type = state[index]
    pairs = count_pairs(state=state, index=index, number_of_cooperators=number_of_cooperators, d=d)
    payoff = payoff_for_number(r=r, alpha=alpha, number_of_cooperators=number_of_cooperators, d=d, type=type)
    return pairs * payoff

for number_of_cooperators in range(1, d+1):
    for index in (0,1):
        for r in (r_high, r_low):
            payoff_combination = get_total_payoff_from_makeup(r=r, alpha=alpha, d=d, index=index, state=state, number_of_cooperators=number_of_cooperators)
            
            type_str = "defector" if state[index] == 0 else "cooperator"
            if index == 1 and number_of_cooperators == 0:
                payoff_combination = 0
            elif index == 0 and number_of_cooperators == d:
                payoff_combination = 0
            data = [[number_of_cooperators, r, type_str, payoff_combination]]
            pd.DataFrame(data=data, columns=columns).to_csv(file_path.parent/"main.csv", mode="a", index=False, header=None)

