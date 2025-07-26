# Libraries
import numpy as np
import matplotlib.pyplot as plt

def lcp_random_number_generator(seed: int, a: int, b: int, c: int, n: int) -> list[int]:
    """
    Generates a list of n pseudo-random numbers using a linear congruential generator (LCG).
    Formula: X_{n+1} = (a * X_n + b) mod c

    Parameters:
        seed (int): Initial value X_0
        a (int): Multiplier (> 0)
        b (int): Increment (≥ 0)
        c (int): Modulus (> 0, ideally a power of 2)
        n (int): Number of values to generate (> 0)

    Returns:
        List of n pseudo-random integers
    """
    
    # Check input validity
    if not (isinstance(a, int) and isinstance(b, int) and isinstance(c, int) and isinstance(seed, int) and isinstance(n, int)):
        raise TypeError("All parameters must be integers.")
    if not (a > 0 and c > 0 and b >= 0 and seed >= 0 and n > 0):
        raise ValueError("Constraints: a > 0, c > 0, b ≥ 0, seed ≥ 0, n > 0")

    res = []
    x = seed

    for _ in range(n):
        x = (a * x + b) % c
        res.append(x)

    return res

def uniform_number_generator(seed: int, a: int, b: int, c: int, n: int) -> list[float]:
    """
    Generates a list of uniformly distributed numbers in the interval [0,1] from the LCP-rgn algorithm generated list.

    Parameters:
        seed (int): Initial value X_0
        a (int): Multiplier (> 0)
        b (int): Increment (≥ 0)
        c (int): Modulus (> 0, ideally a power of 2)
        n (int): Number of values to generate (> 0)
    
    Returns:
        List of uniformly distributed numbers
    """

    sequence = np.array(lcp_random_number_generator(seed=seed, a=a, b=b, c=c, n=n))

    return list(sequence/c)


def n_point_cdf(seq: list[float], values_and_probs : dict[float : float]) -> list[float]:
    """
    From a sequence of uniformly distributed variables, assigns each value from the distribution
    defined by `values_and_probs` according to its cumulative probability.

    Parameters:
        seq (list[float]) : List of Uniform(0,1) values
        values_and_probs (dict[float:float]) : Dictionary mapping discrete values to their probabilities

    Returns:
        List of values sampled from the discrete distribution
    """
    # First, ensure that the probabilities sum to 1
    sum = 0
    for value in values_and_probs.values():
        if value <= 0:
            raise ValueError("Probabilities have to be between 0 and 1")
        sum += value

    if sum != 1:
        raise ValueError(f"Probabilities have to sum to 1 - currently:{sum}")

    # Secondly sort the dictionnary by probability
    values_and_probs = sorted(values_and_probs.items())

    # Generate a matching dict : indeed we want to set X = x_{j+1} if s_j < U <= s_{j+1} where s_j
    # is the sum of all probabilities [i] until [j] and U is the value in seq.

    cdf = []
    cumulative = 0.0

    # Build CDF intervals: [(0.0, 0.1, x1), (0.1, 0.5, x2), ...]
    for value, prob in values_and_probs:
        lower = cumulative
        cumulative += prob
        cdf.append((lower, cumulative, value))
    
    # Assign each U to a value based on where it falls in the CDF
    assigned_values = []

    for U in seq:
        for lower, upper, value in cdf:
            if lower < U <= upper:
                assigned_values.append(value)
                break
    
    return assigned_values

# Utils
def plot_list_numbers(seq:list[float]) -> None:
    plt.figure(figsize=(8,5))
    plt.hist(seq)
    plt.show()

if __name__ == "__main__":
    seed = 42

    # Implementation of the RANDU generator

    RANDU = lcp_random_number_generator(seed = seed, a = 65539, b = 0, c = 2**31, n = 100)
    #print(RANDU)
    
    # Implementation of the IBM System 360 Uniform Random Number Generator
    IBM = lcp_random_number_generator(seed=seed, a = 16807, b = 0, c = 2**31 - 1, n = 100)
    #print(IBM)

    # Generate uniformly distributed list using IBM parameters
    uni_list = uniform_number_generator(seed=seed, a = 16807, b = 0, c = 2**31 - 1, n = 10000)
    #print(uni_list)

    # Plot results
    #plot_list_numbers(uni_list)

    # Generate a two point random variable X
    res = n_point_cdf(seq = uni_list, values_and_probs={'x1':0.3, 'x2':0.7})
    
    plot_list_numbers(res)
