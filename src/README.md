# Maynard

Functionality to generate transition matrices, absorbing states, and other
results for the heterogeneous moran process.

## Tutorial

In this tutorial we will see how to use `maynard` in order to generate the
transition matrix defining the Moran process on a heterogeneous public goods
game. For background
information on Markov processes in general we recommend [1], and for
information on the Moran process and further reading into evolutionary game
theory we recommend [2].

We begin by defining a state space. This is the set of possible populations for
a $N$ players with $k$ actions, and can be obtained by the following code. In this
example, we shall use $N=3$ and $k=2$

```python
import main

N = 3
k = 2

state_space = main.get_state_space(N=N, k=k)
```

which gives

```
array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
])
```

We must then
define the strictly positive fitness function which we will use
to calculate our transition probabilities. The general fitness function for a
heterogeneous public goods game is:

```python
def heterogeneous_contribution_fitness_function(
    state,
    omega,
    r,
    contribution_vector,
    **kwargs
):

    total_goods = (
        r
        * sum(action * contribution for action, contribution in zip(state, contribution_vector))
        / len(state)
    )

    payoff_vector = np.array([total_goods - (action * contribution) for action, contribution in zip(state, contribution_vector)])

    return 1 + (omega * payoff_vector)
```

As the public goods game is a key focus of this library, there is functionality
to calculate a contribution_vector for this function. We will calculate a
linear contribution vector:

```python
contribution_vector = np.array([2,4,6])
```

Note that all supported contribution rules take the parameter M, which is the
sum of all contributions. Now we can pass this, along with the other \*\*kwargs
parameters needed for our fitness function, into our generate_transition_matrix
function.

```python
r=2
omega=0.1

main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=heterogeneous_contribution_fitness_function,
    alpha=contribution_vector,
    r=r
    omega=omega
)
```

which will produce the following transition matrix:

```
array([[1.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.25925926, 0.59259259, 0.        , 0.07407407, 0.        ,
        0.07407407, 0.        , 0.        ],
       [0.24836601, 0.        , 0.58169935, 0.08496732, 0.        ,
        0.        , 0.08496732, 0.        ],
       [0.        , 0.13888889, 0.13888889, 0.52777778, 0.        ,
        0.        , 0.        , 0.19444444],
       [0.23611111, 0.        , 0.        , 0.        , 0.56944444,
        0.09722222, 0.09722222, 0.        ],
       [0.        , 0.13450292, 0.        , 0.        , 0.13450292,
        0.53216374, 0.        , 0.19883041],
       [0.        , 0.        , 0.12962963, 0.        , 0.12962963,
        0.        , 0.53703704, 0.2037037 ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 1.        ]])
```

## How to guides

### How to compute the absorption probabilities of a transition matrix

Given a transition matrix $T$ for a state space $S$, we want to find the
probability of the chain being absorbed into each absorbing state. To do this,
we use the generate_absorption_matrix function. First, define $T$. Here we will
use the transition matrix from the tutorial section.

```python
import numpy as np
import main

transition_matrix = np.array([
    [1.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        ],
    [0.25925926, 0.59259259, 0.        , 0.07407407, 0.        ,
    0.07407407, 0.        , 0.        ],
    [0.24836601, 0.        , 0.58169935, 0.08496732, 0.        ,
    0.        , 0.08496732, 0.        ],
    [0.        , 0.13888889, 0.13888889, 0.52777778, 0.        ,
    0.        , 0.        , 0.19444444],
    [0.23611111, 0.        , 0.        , 0.        , 0.56944444,
    0.09722222, 0.09722222, 0.        ],
    [0.        , 0.13450292, 0.        , 0.        , 0.13450292,
    0.53216374, 0.        , 0.19883041],
    [0.        , 0.        , 0.12962963, 0.        , 0.12962963,
    0.        , 0.53703704, 0.2037037 ],
    [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 1.        ]
    ])
```

Now we want to run the generate_absorption_matrix function. This will return
the absorption matrix for this Moran process.

```python
main.generate_absorption_matrix(transition_matrix=transition_matrix)
```

which will return

```
array([[0.80145592, 0.19854408],
       [0.77428822, 0.22571178],
       [0.46345416, 0.53654584],
       [0.74481676, 0.25518324],
       [0.44455339, 0.55544661],
       [0.42534939, 0.57465061]])
```

### How to compute symbolic transition and absorption matrices

`Maynard` supports sympy.Symbol values. Begin by defining a state
space as before, but now consider a fitness function that gives symbolic
values.

```python
import main
import sympy as sym
import numpy as np

state_space = main.get_state_space(N=3, k=2)

def symbolic_fitness_function(state, **kwargs):
    """
    returns x for an individual of type 1, and y for an individual of type
    0
    """
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )
```

We compute our transition matrix as before

```python
symbolic_transition_matrix = main.generate_transition_matrix(
    fitness_function=symbolic_fitness_function,
    state_space=state_space
)
sym.Matrix(symbolic_transition_matrix)
```

which returns the following sympy.Matrix:

$$
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\[6pt]
\frac{2y}{3x + 6y} &
1 - \frac{2x}{3x + 6y} - \frac{2y}{3x + 6y} &
0 &
\frac{x}{3x + 6y} &
0 &
\frac{x}{3x + 6y} &
0 &
0 \\[6pt]
\frac{2y}{3x + 6y} &
0 &
1 - \frac{2x}{3x + 6y} - \frac{2y}{3x + 6y} &
\frac{x}{3x + 6y} &
0 &
0 &
\frac{x}{3x + 6y} &
0 \\[6pt]
0 &
\frac{y}{6x + 3y} &
\frac{y}{6x + 3y} &
1 - \frac{2x}{6x + 3y} - \frac{2y}{6x + 3y} &
0 &
0 &
0 &
\frac{2x}{6x + 3y} \\[6pt]
\frac{2y}{3x + 6y} &
0 &
0 &
0 &
1 - \frac{2x}{3x + 6y} - \frac{2y}{3x + 6y} &
\frac{x}{3x + 6y} &
\frac{x}{3x + 6y} &
0 \\[6pt]
0 &
\frac{y}{6x + 3y} &
0 &
0 &
\frac{y}{6x + 3y} &
1 - \frac{2x}{6x + 3y} - \frac{2y}{6x + 3y} &
0 &
\frac{2x}{6x + 3y} \\[6pt]
0 &
0 &
\frac{y}{6x + 3y} &
0 &
\frac{y}{6x + 3y} &
0 &
1 - \frac{2x}{6x + 3y} - \frac{2y}{6x + 3y} &
\frac{2x}{6x + 3y} \\[6pt]
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

Now we can generate a symbolic absorption matrix:

```python
generate_absorption_matrix(symbolic_transition_matrix, symbolic=True)
```

which generates a sympy.Matrix:

$$
\left[\begin{matrix}\frac{4 y \left(3 x^{3} y + 9 x^{2} y^{2} + 6 x y^{3}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{2 y \left(12 x^{4} + 51 x^{3} y + 81 x^{2} y^{2} + 66 x y^{3} + 24 y^{4}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} & \frac{2 x \left(6 x^{3} y + 3 x^{2} y^{2}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{4 x \left(12 x^{4} + 24 x^{3} y + 21 x^{2} y^{2} + 6 x y^{3}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)}\\\frac{4 y \left(3 x^{3} y + 9 x^{2} y^{2} + 6 x y^{3}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{2 y \left(12 x^{4} + 51 x^{3} y + 81 x^{2} y^{2} + 66 x y^{3} + 24 y^{4}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} & \frac{2 x \left(6 x^{3} y + 3 x^{2} y^{2}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{4 x \left(12 x^{4} + 24 x^{3} y + 21 x^{2} y^{2} + 6 x y^{3}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)}\\\frac{2 y \left(3 x^{2} y^{2} + 6 x y^{3}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{4 y \left(6 x^{3} y + 21 x^{2} y^{2} + 24 x y^{3} + 12 y^{4}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} & \frac{4 x \left(6 x^{3} y + 9 x^{2} y^{2} + 3 x y^{3}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{2 x \left(24 x^{4} + 66 x^{3} y + 81 x^{2} y^{2} + 51 x y^{3} + 12 y^{4}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)}\\\frac{4 y \left(3 x^{3} y + 9 x^{2} y^{2} + 6 x y^{3}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{2 y \left(12 x^{4} + 51 x^{3} y + 81 x^{2} y^{2} + 66 x y^{3} + 24 y^{4}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} & \frac{2 x \left(6 x^{3} y + 3 x^{2} y^{2}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{4 x \left(12 x^{4} + 24 x^{3} y + 21 x^{2} y^{2} + 6 x y^{3}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)}\\\frac{2 y \left(3 x^{2} y^{2} + 6 x y^{3}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{4 y \left(6 x^{3} y + 21 x^{2} y^{2} + 24 x y^{3} + 12 y^{4}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} & \frac{4 x \left(6 x^{3} y + 9 x^{2} y^{2} + 3 x y^{3}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{2 x \left(24 x^{4} + 66 x^{3} y + 81 x^{2} y^{2} + 51 x y^{3} + 12 y^{4}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)}\\\frac{2 y \left(3 x^{2} y^{2} + 6 x y^{3}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{4 y \left(6 x^{3} y + 21 x^{2} y^{2} + 24 x y^{3} + 12 y^{4}\right)}{\left(3 x + 6 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} & \frac{4 x \left(6 x^{3} y + 9 x^{2} y^{2} + 3 x y^{3}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)} + \frac{2 x \left(24 x^{4} + 66 x^{3} y + 81 x^{2} y^{2} + 51 x y^{3} + 12 y^{4}\right)}{\left(6 x + 3 y\right) \left(8 x^{4} + 22 x^{3} y + 30 x^{2} y^{2} + 22 x y^{3} + 8 y^{4}\right)}
\end{matrix}\right]
$$

### How to write a fitness function

A fitness function is a function
which takes a state and returns an ordered numpy.array of strictly positive
floats. The general form of a fitness function would be as follows:

- Take state and any \*\*kwargs
- Calculate the fitness of each individual
- Return an ordered array of fitness values for each player

An example for the two-thirds game [5] would be as follows:

```python
def example_fitness_function(state, **kwargs):

    avg = sum(state) / len(state)
    target = (2/3) * avg

    return np.array([abs(state[i] - target) for i in state])
```

### How to write a contribution rule

A contribution rule for the get_contribution_vector function is a function
which takes a player's index and the number of players (index, N), and returns
the contribution for that player. This function is specifically for modelling
the public goods game. The general form of a contribution rule would be as
follows:

- Take a player's index and the total number of players, along with any \*\*kwargs
- Calculate that player's contribution
- Return that player's contribution

Contribution rules generally take a number of \*\*kwargs in order to perform
their calculations, however index and N are the only required parameters for
such a function.

An example to return a contribution equal to your index multiplied by the
number of players would be:

```python
def example_contribution_rule(index, N, **kwargs):

    return index * N
```

### How to generate a contribution vector based on a distribution

There is functionality to generate contribution vectors based on a Dirichlet
distribution (for further reading, see [6]).
For this, we use the following function:

```python
import main
import contribution_rules
N=3
M=12

main.get_dirichlet_contribution_vector(
    N=N,
    alpha_rule=contribution_rules.dirichlet_log_alpha_rule,
    M=M)
```

which returns

```
array([2.54202934, 4.16458154, 5.29338912])
```

The returned array will always sum to M.

### How to write an alpha rule

An alpha rule will take a value N and return a numpy.array of N
strictly positive floats.

An example returning a sequence of increasing alpha parameters would be:

```python
def example_alpha_rule(N, **kwargs):

    return np.array([i for i in range(N)])
```

## Explanation

### A brief explanation of the Moran process

A Moran process is a type of Markov process often used in evolutionary game
theory. We define a set of $N$ players, each playing one of $k$ actions, $A_1,
A_2, ..., A_k$. We denote the set of all possible actions as $\textbf{K}$.
We then define a state space,
$S$, which is an ordered set of ordered N-tuples with entries in $\textbf{K}$.
These states represent all possible permutations of each
player performing each action.

For example, the state space for $N=3$ and $k=2$ is:

```
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
```

Now we define a fitness function which assigns a non-zero value to each
player in a given state $\textbf{v}$. A higher fitness represents a player who is
performing "better" in a state. An example of this would be the function we use
for the homogeneous public goods game with contribution $\alpha$ and selection
intensity $\omega$ (in order to ensure positivity).

$$
f: \textbf{K}^N \to \mathbb{R}^N \\

f(\textbf{v})_i = 1 + \omega(\frac{ \sum_{j=1}^N (\alpha v_j)}{N} - \alpha v_i)
$$

Then we calculate transition probabilities according to the following formula:

$$
\large
 p_{v, u} =
        \begin{cases}
            \frac{\sum_{v_i = u_{i^*}}{f(v_i)}}{\sum_{v_i}f(v_i)} & \text{if }h(v,u) = 1 \text{, differing at position }i^*\\
            0 & \text{if }h(v,u) > 1\\
            1 - \sum_{u \in S \setminus \text{\{v\}}}{P_{v,u}} & \text{if }h(v,u) = 0
        \end{cases}
$$

These probabilities form the entries of our transition matrix $T$. In an
ordered state space, the entry $T_{i,j}$ represents the probability of
transitioning from the state at index $i$ to the state at index $j$.

In the Moran process, transitioning from one state to another represents one
individual changing their action type from their own to that of another individual in the
population. The process enters an absorbing state
(one which can only transition to itself) when all individuals are of the same
type, as there is no alternative type of individual to imitate. Transitions
occur with a probability proportional to the fitness of each individual.

### Finding the absorption matrix of an absorbing Markov chain.

Consider some transition matrix $T$. Then we can define the following
submatrices:

- $Q$: The submatrix of probabilities of transitioning between transitive
  states (non-absorbing states)
- $R$: The submatrix of probabilities of transitioning from a transitive state
  to a non-transitive state.

Now we can define the _fundamental matrix_ of a Markov chain by the
equation $F = (I-Q)^{-1}$. $F_{ij}$ corresponds to the expected number of times
the chain will be in state $j$ given that it started in state $i$.

Now we calculate our _absorption matrix_ $B$ by the formula $B = FR$.

$B$ will be an $(N-k)$ x $k$ matrix,
as we have exactly $k$ absorbing states; those where each individual is of the
same type.

### How to interpret an absorption matrix

Consider a state space

```
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
```

We can split this into two arrays: one for the transitive states, and one for
the absorbing states:

```
Transitive states:
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],

Absorbing states:
    [0,0,0],
    [1,1,1]
```

Now, the entry $i,j$ in the absorption matrix is the probability of ending up
in absorbing state $j$ after beginning in transitive state $i$. So entry (1,2)
is the probability of ending up in state [1,1,1] after beginning in [0,0,1].

### The affect of $\alpha$ parameters on the Dirichlet distribution

There are three key factors which affect the
realisation based on the $\alpha$ parameters.

- The number of $\alpha$ parameters determines the number of entries in the realisation
  vector

- The ratio between the $\alpha$ parameters will be reflected in the realisation. For
  example, if we have $\alpha_1 = 1, \alpha_2 = 2, \alpha_3 = 3$ then the
  expected value of our realisation will be
  $(\frac{1}{6}, \frac{2}{6}, \frac{3}{6})$. This is because the expected value
  of entry $i$ in our realisation is $\tilde{\alpha}_i =
  \frac{\alpha_i}{\sum_{j=1}^{N}\alpha_j}$

- $\alpha_0 = \sum_{i=1}^{N} \alpha_i$ determines the standard deviation of
  each entry in
  the realisation. This is because the variance of each component is given by
  $\frac{\tilde{\alpha}_i (1 - \tilde{\alpha}_i)}{\alpha_0 + 1}$

For further reading on the derivation of these results, see [6]

### Transforming a realisation of the Dirichlet distribution into a contribution vector

Once we obtain a realisation of the Dirichlet distribution, we use it to
produce a contribution vector. This is done by multiplying the realisation by
some parameter $M$, defining the total contribution of all individuals. This is
the only necessary step as for any realisation of the Dirichlet distribution
$X = (X_1, X_2, ...)$, we have $\sum_{i=1}^{N} X_i = 1$. This is because a
Dirichlet random vector is constructed by sampling independent Gamma
distributions $Y_i \sim \operatorname{Gamma}(\alpha_i, 1)$ and normalising to
$X_i = \frac{Y_i}{\sum_{j=1}^{K} Y_j}$, which then sums to 1 across the $X_is$.
[6]

## References

### List of functionality

The following functions are available in main:

- `get_state_space`
- `compute_transition_probability`
- `generate_transition_matrix`
- `get_absorbing_state_index`
- `get_absorbing_states`
- `extract_Q`
- `extract_R_numerical`
- `extract_R_symbolic`
- `generate_absorption_matrix_numerical`
- `generate_absorption_matrix_symbolic`
- `generate_absorption_matrix`

The following functions are available in contribution_rules:

- `dirichlet_linear_alpha_rule`
- `dirichlet_binomial_alpha_rule`
- `dirichlet_log_alpha_rule`
- `log_contribution_rule`
- `linear_contribution_rule`
- `binomial_contribution_rule`

The following functions are depreciated, and more up to date alternatives
are available:

- `get_absorption_probabilities`

### Bibliography

The wikipedia pages on absorbing Markov chains and Moran processes give good
overviews of the topics:

[1] <https://en.wikipedia.org/wiki/Absorbing_Markov_chain>,

[2] <https://en.wikipedia.org/wiki/Moran_process>

The following textbook is also a very good resource and reference on Markov
chains

> [3] Stewart, William J. Probability, Markov chains, queues, and simulation: the
> mathematical basis of performance modelling. Princeton university press, 2009.

To read more about evolutionary game theory, the following textbook is well
regarded

> [4] Martin A. Nowak. Evolutionary Dynamics: Exploring the Equations of Life

For information on the $\frac{2}{3}$ game:

[5] https://en.wikipedia.org/wiki/Guess_2/3_of_the_average

For further reading into the Dirichlet Distribution, see

> [6] Jiayu Lin. On the Dirichlet Distribution. Queen’s University
> Kingston, Ontario, Canada September 2016
