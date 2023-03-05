from itertools import permutations, product
from functools import partial
from fractions import Fraction
import numpy as np
import scipy.sparse as sps
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import toeplitz
import networkx as nx

from constants import *


# ===== GENERATING METRIC SPACES =====

def gen_graph(n, graph_type=ERDOS, enforce_n=False, rnd=None, **kwargs):
    """
    Generate adjacency and distance matrices of a random graph with (up to) n vertices.
    """
    rnd = rnd or np.random.RandomState(SEED)

    if graph_type == ERDOS:
        gen_G = partial(nx.erdos_renyi_graph, n=n, p=kwargs['p'], seed=rnd)
    elif graph_type == WATTS:
        gen_G = partial(nx.watts_strogatz_graph, n=n, k=2*kwargs['k'], p=kwargs['p'], seed=rnd)
    elif graph_type == BARABASI:
        gen_G = partial(nx.barabasi_albert_graph, n=n, m=kwargs['m'], seed=rnd)
    else:
        raise ValueError(f'Unknown graph type {graph_type}')

    while True:
        G = gen_G()
        largest_connected_comp = list(max(nx.connected_components(G), key=len))
        if len(largest_connected_comp) == n or not enforce_n:
            A = nx.to_numpy_array(G)[np.ix_(largest_connected_comp, largest_connected_comp)]
            break

    X = sps.csgraph.shortest_path(A, directed=False, unweighted=True)

    return X


def gen_point_cloud(n, metric='euclidean', rnd=None):
    """
    Generates distance matrix of n points in R^(n-1).
    """
    rnd = rnd or np.random.RandomState(SEED)

    points = rnd.rand(n, n - 1)
    distances = pdist(points, metric=metric)
    X = squareform(distances)

    return X


def gen_interval(n):
    """
    Generates distance matrix of unweighted path graph on n vertices.
    """
    X = toeplitz(np.arange(n))

    return X


def gen_simplex(n):
    """
    Generates regular simplex with n vertices with unit side.
    """
    X = np.ones((n, n))
    np.fill_diagonal(X, 0)

    return X


# ===== METRIC SPACE PROPERTIES =====

def check_triangle(X, ultra=False, verbose=True):
    """
    Checks if a distance matrix satisfies the (possibly strong) triangle inequality.
    """
    assert np.all(X == X.T)
    for i, j, k in permutations(range(len(X)), 3):
        rhs = max(X[j, k], X[k, i]) if ultra else X[j, k] + X[k, i]
        try:
            assert X[i, j] <= rhs
        except AssertionError:
            if verbose:
                rhs_symbols = f'max(d{j}{k}, d{k}{i})' if ultra else f'd{j}{k} + d{k}{i}'
                rhs_numbers = f'max({X[j, k]}, {X[k, i]})' if ultra else f'{X[j, k]} + {X[k, i]}'
                print(f'NOT A METRIC SPACE: d{i}{j} > {rhs_symbols}: {X[i, j]} > {rhs_numbers}')
            return False

    return True


def card(X):
    return X if isinstance(X, int) else len(X)


def diam(X):
    return X.max()


def rad(X):
    return np.min(X.max(axis=0))


def permute(X, rnd=None):
    """
    Permutes the order of points in X.

    :param X: distance matrix of X
    :param rnd: NumPy random state
    :return: distance matrix of permuted X, 2d-array
    """
    rnd = rnd or np.random.RandomState(SEED)

    f = rnd.permutation(card(X))
    permuted_X = X[np.ix_(f, f)]

    return permuted_X

# ===== MAPPINGS =====

def gen_fs(X, Y, injective=False):
    """
    Generates all possible mappings in X🠖Y.

    :param X: distance matrix of X or |X|
    :param Y: distance matrix of Y or |Y|
    :param injective: whether the mappings should be injective
    :return: f:X🠖Y, 1d-array
    """
    n, m = card(X), card(Y)
    for f in (permutations(range(m), n) if injective else product(range(m), repeat=n)):
        yield np.array(f)


def gen_Fs(X, Y, injective=False):
    """
    Generates all possible mappings in X🠖Y as {0,1}^|X|×|Y|.

    :param X: distance matrix of X or |X|
    :param Y: distance matrix of Y or |Y|
    :param injective: whether the mappings should be injective
    :return: F, 2d-array
    """
    for f in gen_fs(X, Y, injective=injective):
        yield f_to_F(f)


def rnd_f(X, Y, injective=False, rnd=None):
    """
    Generates random mapping in X🠖Y.

    :param X: distance matrix of X or |X|
    :param Y: distance matrix of Y or |Y|
    :param injective: whether the mappings should be injective
    :param rnd: NumPy random state
    :return: mapping, 1d-array
    """
    rnd = rnd or np.random.RandomState(SEED)

    n, m = card(X), card(Y)
    f = rnd.choice(m, n, replace=not injective)

    return f


def rnd_P(X, Y, rnd=None):
    """
    Generates random soft mapping in X🠖Y as a point in the mapping polytope 𝒫 ⊂ [0,1]|X|×|Y|.

    :param X: distance matrix of X or |X|
    :param Y: distance matrix of Y or |Y|
    :param rnd: NumPy random state
    :return: soft mapping, 2d-array
    """
    rnd = rnd or np.random.RandomState(SEED)

    P = rnd.rand(card(X), card(Y))
    P /= P.sum(axis=1)[:, None]

    return P


def central_P(X, Y):
    """
    Returns soft mapping that is the barycenter of the mapping polytope 𝒫 ⊂ [0,1]|X|×|Y|.

    :param X: distance matrix of X or |X|
    :param Y: distance matrix of Y or |Y|
    :return: soft mapping, 2d-array
    """
    n, m = card(X), card(Y)
    P = np.full((n, m), 1 / m)

    return P


def f_to_F(f, Y):
    """
    Converts mapping representation {1,…,|Y|}^|X| to {0,1}^|X|×|Y|.
    """
    m = card(Y)
    F = np.identity(m)[f]

    return F


def P_to_f(P):
    """
    Projects soft mapping in [0,1]^|X|×|Y| onto the space of mappings
    {0,1}^|X|×|Y|, represented as {1,…,|Y|}^|X|.
    """
    f = np.argmax(P, axis=1)

    return f


def project_P(P):
    """
    Projects soft mapping in [0,1]^|X|×|Y| onto the space of mappingṡ {0,1}^|X|×|Y|.
    """
    f = P_to_f(P)
    F = f_to_F(f, P.shape[1])

    return F


def dis(P, X, Y):
    """
    Calculates soft distortion, which coincides with distortion whenever the
    soft mapping is in {0,1}^|X|×|Y|.

    :param P: point in the mapping polygon, matrix or vector
    """
    if P.ndim == 1:
        P = f_to_F(P, Y)

    dis_P = np.abs(X - P @ Y @ P.T).max()

    return dis_P


def l2dis(P, X, Y):
    """
    Calculates soft (squared) l2-distortion (l∞-norm is replaced with l2-norm),
    which coincides with distortion whenever the soft mapping is in {0,1}^|X|×|Y|.

    :param P: row-stochastic matrix in [0,1]^|X|×|Y|
    :param X: distance matrix of X
    :param Y: distance matrix of Y
    :return: l2-distortion of P
    """
    sq_l2dis_P = ((X - P @ Y @ P.T)**2).sum()

    return sq_l2dis_P


def f_g_to_R(f, g):
    """
    Constructs correspondence out of mapping pair f:X🠖Y, g:Y🠖X.
    """
    return set((*enumerate(f), *((x, y) for y, x in enumerate(g))))


# ===== METRIC SPACE REPRESENTATIONS =====

def A_to_K(A, t=1):
    """
    Converts adjacency matrix to heat transfer matrix.

    :param A: adjacency matrix
    :param t: time parameter (how long the heat dissipates)
    :return: heat transfer matrix (how much heat transferred between each
        pair of points after time t)
    """
    lambdas, Phi = la.eigh(sps.csgraph.laplacian(A, normed=True))

    K = Phi @ np.diagflat(np.exp(-t * lambdas)) @ Phi.T

    assert np.allclose(K, K.T), 'non-symmetric heat transfer matrix'
    assert np.all(la.eigvalsh(K) > 0), 'non-positive eigenvalues of heat transfer matrix'

    return K


def X_to_K(X, t=1, type='exp_t'):
    """
    Converts distance matrix to heat transfer matrix by representing the space
    as complete graph.

    :param X: distance matrix
    :param t: time parameter
    :param type: how to obtain edge weights from distances
    :return: heat transfer matrix (how much heat transferred between each
        pair of points after time t)
    """
    if type == 'exp':
        variance = np.var(X[np.triu_indices_from(X, k=1)])
        A = np.e ** (-X ** 2 / variance)
    elif type == 'exp_t':
        A = np.e ** (-X ** 2 / 4 / t)
    elif type == 'inv':
        A = 1/X
    np.fill_diagonal(A, 0)

    return A_to_K(A, t=t)


# ===== MISC =====

def get_fraction(x, max_denom=100):
    return Fraction(x).limit_denominator(max_denom)

