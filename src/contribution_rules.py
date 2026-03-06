import scipy
import sympy as sym
import numpy as np
import scipy.special
import math


def dirichlet_linear_alpha_rule(N):
    """
    Generates the alphas for a population contributing according to a linear
    rule in the dirichlet distribution. Each player's alpha is equal to their
    position in the state (indexed from 1). Therefore, as we take the mean
    across many realisations, we get that player i will contribute
    i * (contribution of player 1).

    Parameters
    -----------
    N - int, the number of players in a state

    returns
    ---------
    numpy.array - a linear counting from 1 to N."""

    return np.array([i + 1 for i in range(N)])


def dirichlet_binomial_alpha_rule(N, n, low_alpha, high_alpha, **kwargs):
    """
    Generates the alphas for a population contributing according to a binomial
    rule in the dirichlet distribution. We have a low alpha and a high alpha. n
    players contribute according to the low alpha, and N-n contribute according
    to the high alpha. These form two distinct sets of components for the
    vector obtained via the dirichlet distribution - those with a low mean and those with a high mean.

    Parameters
    -----------
    N - int, the number of players in a state

    n - int, the number of players who contribute according to the low alpha

    low_alpha - float, a strictly positive value for the low mean cooperators to
    contribute according to. Less than high_alpha.

    high_alpha - float, a strictly positive value for the high mean cooperators
    to contribute according to. Greater than low_alpha

    returns
    ---------
    numpy.array - a bipartite set containing n low alphas and N-n high alphas"""

    return [low_alpha for _ in range(n)] + [high_alpha for _ in range(N - n)]


def dirichlet_log_alpha_rule(N):
    """
    Generates the alphas for a population contributing according to a
    logarithmic transformation of the linear alpha rule in the dirichlet
    distribution. Each player's alpha is equal to the natural log of their
    position in the state +1 (in order to ensure strict positivity among the
    alphas, since otherwise player 1 would have $\alpha = 0$). Therefore, as we
    take the mean across many realisations, we get a logarithmic scale where
    the mean contribution for each player is always greater than the last, but
    the difference decreases as N increases.

    Parameters
    -----------
    N - int, the number of players in a state

    returns
    ---------
    numpy.array - the alpha value of each player according to their index. ."""

    return np.array([np.log(i) + 1 for i in range(1, N + 1)])


def dirichlet_power_law_alpha_rule(N, a=np.exp(1)):
    """
    Generates the alphas for a population contributing according to a power law
    transformation of the linear alpha rule in the dirichlet distribution. Each
    player's alpha is equal to $a^i$, with $a = e$ by default, where $i$ is
    their index in the state (indexed from 1). Therefore, as we take the mean
    across many realisations, we get an exponential scale where the mean
    contribution for each player increases exponentially (though not
    necessarily using the exponential $e$)

    Parameters
    -----------
    N - int, the number of players in a state

    a - float, the base of the power law (the number which is raised to the
    power i), Must be strictly positive, and is taken as $e$ as standard.

    returns:
    --------
    numpy.array - the alpha value of each player according to their index"""

    if a <= 0:
        raise ValueError("a must be strictly positive")

    return np.array([a ** (i) for i in range(1, N + 1)])


def log_contribution_rule(index, M, N):
    """
    Players contribute according to a logarithmic scale of a linear
    contribution rule. This corresponds to the equation
    $\sum_{i=1}^{N} log(\lambda i) = M$. Here, we calculate a lambda value
    $(\frac{e^M}{N!})^{1/N}$ (proof in main.tex)

    Parameters:
    ------------
    index: integer, the position of the player within the ordered population


    M: float, the population maximum contribution

    N: integer, the size of the population

    returns:
    ---------
    float, the contribution of the player at (index) according to a logarithmic
    contribution rule when contributing"""

    return sym.log((index + 1) * ((sym.exp(M) / scipy.special.factorial(N)) ** (1 / N)))


def linear_contribution_rule(index, N, M):
    """
    Players contribute according to a linear contribution rule. As stated in main.tex, this corresponds to the equation
    $\sum_{i=1}^{N} \lambda i = M$. Here, we calculate a lambda value
    $\frac{2M}{N(N+1)}$ (proof in main.tex)

    Parameters:
    ------------
    index: integer, the position of the player within the ordered population

    M: float, the population maximum contribution

    N: integer, the size of the population

    returns:
    ---------
    float, the contribution of the player at (index) according to a linear
    contribution rule when contributing"""

    K = (2 * M) / (N * (N + 1))

    return (index + 1) * K


def binomial_contribution_rule(index, N, n, M, alpha_h):
    """
    Players contribute according to a binomial contribution rule, where n
    players contribute a low amount, and N-n contribute a high amount. As stated in main.tex, this corresponds to the equation
    $n \alpha_l + (N-n)\alpha_h = N\alpha_h - nd = M$, where d is the
    difference between low and high contributions.

    Parameters:
    ------------
    index: integer, the position of the player within the ordered population

    M: float, the population maximum contribution

    N: integer, the size of the population

    alpha_h: float, the "high" contribution according to the binomial
    contribution rule

    n: integer, the number of low contributing players.

    returns:
    ---------
    float, the contribution of the player at (index) according to a binomial
    contribution rule for a given alpha_h when contributing"""

    d = (N * alpha_h - M) / (n)
    alpha_l = alpha_h - d
    if alpha_l < 0:
        raise ValueError("alpha_l is negative with these parameters")
    if alpha_l > alpha_h:
        raise ValueError("alpha_l is greater than alpha_h with these parameters")

    return alpha_l if index < n else alpha_h


def power_law_contribution_rule(index, N, M, a=np.exp(1)):
    """
    Players contribute according to a power law contribution rule. This
    corresponds to the equation $\sum_{i=1}^{N} \lambda a^{i} = M$, where we
    obtain $\lambda = log_a(\frac{M}{$\sum_{i=1}^{N} a^i})$. As standard, we
    take a = e, however this can be taken as any positive value.

    Parameters:
    ------------
    index: integer, the position of the player within the ordered population

    M: float, the population maximum contribution

    N: integer, the size of the population

    a: float, the base of the value of lambda. We take this as $e$ as standard

    returns:
    ---------
    float, the contribution of a player at (index) according to a power law
    contribution rule when contributing."""

    if a <= 0:
        raise ValueError("a must be positive")

    summation_term = np.sum(a ** (i + 1) for i in range(N))

    K = M / summation_term

    return (a ** (index + 1)) * K
