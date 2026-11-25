import ludics
import numpy as np

def generator_for(seed, stream=1):
    """
    Returns a generator for one named stream of a seed.

    Seeding the global generator inside each function makes the draws of
    different functions share a stream. With the same seed the choice
    intensities come out as the very same numbers as the first entries of the
    selection intensities, and the costs are drawn from that stream too. Giving
    each quantity its own stream keeps them independent of one another while the
    seed still fixes all of them.

    Parameters
    ----------
    seed: int, the seed of the population

    stream: int, which quantity is being drawn

    returns
    -------
    numpy.random.Generator: the generator of that quantity"""

    return np.random.default_rng(np.random.SeedSequence([seed, stream]))

def subgroup_pgg_fitness_function(state, r, contribution_vector, d, generator, **kwargs):
    """
    Returns a payoff by actually simulating (realising) the subgroup and having
    players within the subgroup. Sets the fitness of players which do not
    participate to 0.
    
    Parameters
    ----------

    state: numpy.array, the state where 1 is C and 0 is D
     
    r: float, the return on investment

    contribution_vector: numpy.array, the contribution of each player

    d: int, the group size. Must be less than or equal to N

    generator: numpy.random.generator, the generator of the random selection

    returns
    -------
    numpy.array, the payoff of each player in the game
    """
    game_indices = generator.choice(len(state), d, replace=False)

    total_contribution = r/d * np.sum(np.array([contribution_vector[index]*state[index] for index in game_indices]))

    player_to_fitness = {index:total_contribution - contribution_vector[index]*state[index] for index in game_indices}

    return player_to_fitness

@ludics.changes_one_individual
def moran_transition_probability(source, target, selection_intensity, fitness_function=subgroup_pgg_fitness_function, **kwargs):
    """
    Returns the probability of transitioning between two states by the Moran
    process when only players in the subgame can change action type
    
    Parameters
    ----------
    source: numpy.array, the state being transitioned away from
    
    target: numpy.array, the state being transitioned to
    
    selection_intensity: float, the selection intensity of the dynamic

    fitness_function: func, the fitness function
    
    return
    ------
    float, the probability of transitioning between two states"""

    different_indices = np.where(source!=target)[0]
    if len(different_indices) > 1:
        return 0
    elif len(different_indices) == 0:
        return None

    player_to_fitness = fitness_function(state=source, **kwargs)
    index_of_difference = different_indices[0]

    if index_of_difference not in player_to_fitness.keys():
        return 0

    numerator = sum(1 - selection_intensity + selection_intensity * j for i,j in player_to_fitness.items() if source[i] == target[index_of_difference])
    denominator = sum(1 - selection_intensity + selection_intensity * j for j in player_to_fitness.values())
    return 1/len(player_to_fitness.keys()) * numerator/denominator

@ludics.changes_one_individual
def fermi_transition_probability(source, target, choice_intensity, fitness_function=subgroup_pgg_fitness_function, **kwargs):
    """
    Returns the probability of transitioning between two states by Fermi
    imitation dynamics in a subgame pgg where only players who play will switch
    action.
    
    Parameters
    ----------
    source: numpy.array, the state being transitioned away from
    
    target: numpy.array, the state being transitioned to
    
    choice_intensity: float, the choice intensity of the dynamic

    fitness_function: func, the fitness function
    
    return
    ------
    float, the probability of transitioning between two states"""

    different_indices = np.where(source!=target)[0]
    if len(different_indices) > 1:
        return 0
    elif len(different_indices) == 0:
        return None

    player_to_fitness = fitness_function(state=source, **kwargs)
    index_of_difference = different_indices[0]
    if index_of_difference not in player_to_fitness.keys():
        return 0

    changes = [
        ludics.fermi_imitation_function(
            delta=player_to_fitness[index_of_difference] - j,
            choice_intensity=choice_intensity,
            **kwargs,
        )
        for i,j in player_to_fitness.items() if source[i] == target[index_of_difference]
    ]

    scalar = 1 / (len(player_to_fitness.keys()) * (len(player_to_fitness.keys()) - 1))
    return (scalar * np.array(changes)).sum()

def wright_fisher_transition_probability(source, target, selection_intensity, mu, fitness_function=subgroup_pgg_fitness_function, **kwargs):
    """
    Returns the probability of transitioning between two states by a
    Wright-Fisher process in a subgame pgg where only players who play will
    switch action.
    
    Parameters
    ----------
    source: numpy.array, the state being transitioned away from
    
    target: numpy.array, the state being transitioned to
    
    selection_intensity: float, the selection intensity of the dynamic

    mu: float, the mutation probability

    fitness_function: func, the fitness function
    
    return
    ------
    float, the probability of transitioning between two states
    """

    different_indices = ludics.get_different_indices(source=source, target=target)
    if len(different_indices) == 0:
        return None

    player_to_fitness = fitness_function(state=source, **kwargs)
    denominator = sum(1 - selection_intensity + selection_intensity * j for j in player_to_fitness.values())
    transition_dict = {
        0:(sum(1 - selection_intensity + selection_intensity * j for i,j in player_to_fitness.items() if source[i]==0)/denominator)*(1 - 2*mu) + mu,
        1:(sum(1 - selection_intensity + selection_intensity * j for i,j in player_to_fitness.items() if source[i]==1)/denominator)*(1 - 2*mu) + mu
        }
    transition_prob = 1
    for player_index in player_to_fitness.keys(): 
        new_action = target[player_index]
        if player_index not in player_to_fitness.keys() and source[player_index] != target[player_index]:
            transition_prob = 0
        if new_action not in source and mu == 0:
            transition_prob = 0 
        transition_prob = transition_prob * transition_dict[new_action]
    return transition_prob

@ludics.changes_one_individual
def introspection_transition_probability(source, target, choice_intensity, d, r, contribution_vector, fitness_function=subgroup_pgg_fitness_function, **kwargs):
    """
    Returns the probability of transitioning between two states by introspection
    dynamics in a subgame pgg where only players who play will switch action.
    
    Parameters
    ----------
    source: numpy.array, the state being transitioned away from
    
    target: numpy.array, the state being transitioned to
    
    choice_intensity: float, the choice intensity of the dynamic

    d: int, the group size

    r: float, the return on investment

    contribution_vector: numpy.array, the contributions of each player
    
    return
    ------
    float, the probability of transitioning between two states"""

    different_indices = np.where(source!=target)[0]
    if len(different_indices) > 1:
        return 0
    elif len(different_indices) == 0:
        return None

    index_of_difference = different_indices[0]
    player_to_fitness = fitness_function(state=source, d=d, r=r, contribution_vector=contribution_vector, **kwargs)
    if index_of_difference not in player_to_fitness.keys():
        return 0
    transition_difference = contribution_vector[index_of_difference] * (1 - r/d)
    if source[index_of_difference] == 1:
        transition_difference = -1 * transition_difference

    return 1/d * ludics.fermi_imitation_function(delta=transition_difference, choice_intensity=choice_intensity)

def aspiration_transition_probability(source, target, choice_intensity, aspiration,fitness_function=subgroup_pgg_fitness_function, **kwargs):
    """
    Returns the probability of transitioning between two states by aspiration
    dynamics in a subgame pgg where only players who play will switch action.
    
    Parameters
    ----------
    source: numpy.array, the state being transitioned away from
    
    target: numpy.array, the state being transitioned to
    
    choice_intensity: float, the choice intensity of the dynamic

    aspiration: float, the aspiration level
        
    fitness_function: func, the fitness function

    return
    ------
    float, the probability of transitioning between two states"""

    different_indices = np.where(source!=target)[0]
    if len(different_indices) > 1:
        return 0
    elif len(different_indices) == 0:
        return None

    index_of_difference = different_indices[0]
    player_to_fitness = fitness_function(state=source, **kwargs)
    if index_of_difference not in player_to_fitness.keys():
        return 0

    return 1/len(player_to_fitness.keys()) * ludics.fermi_imitation_function(delta=(player_to_fitness[index_of_difference] - aspiration), choice_intensity=choice_intensity)

    