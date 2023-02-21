import numpy as np
from scipy import linalg


def create_prob_simplex(n, m):
    '''
    Create equality constraints for the convex hull of all f:X→Y, i.e.
    the polytope of row-stochastic matrices 𝒫 ⊂ [0,1]|X|×|Y| .
    
    Inputs:
    n: |X| (the size of domain of f)
    m: |Y| (the size of codomain of f)
    
    Outputs:
    A: |X|×|X||Y| coefficient matrix of the equality constraint
    b: |X||Y|×1 right-hand side of the equality constraint
    center: |X|×|Y| barycenter of the probability simplex
    '''
    A = np.kron(np.eye(n), np.ones((1, m)))
    b = np.ones((n, 1))    
    center = np.full((n, m), 1/m)
    
    return A, b, center


def project_onto_prob_simplex(pi, A, b, P, center):
    '''
    Project sampled point onto the polytope of
    row-stochastic matrices of size |X|×|Y|.
    
    Inputs:
    pi: |X|×|Y| matrix representation of sampled point
    A: |X|×|X||Y| coefficient matrix of the equality constraint
    b: |X||Y|×1 right-hand side of the equality constraint
    P: |X||Y|×|X||Y| projection matrix onto the row space of A
    center: |X|×|Y| barycenter of the probability simplex
    
    Outputs:
    projected_pi: |X|×|Y| matrix representation of the projected sampled point
    '''
    n, m = center.shape

    # Create the vector to actually project and reshape.
    vec_to_project = pi - center
    vec_to_project = vec_to_project.reshape(n*m,)

    # Project it.
    vec_to_project -= np.matmul(P, vec_to_project)
    projected_pi = (center.reshape(n*m,) + vec_to_project).reshape(n, m)

    return projected_pi


def markov_hit_and_run_step(pi, A, b, P, center, rng):
    '''
    Sample a new point from the polytope of row-stochastic
    matrices of size |X|×|Y| by a hit-and-run step.
    
    Inputs:
    pi: |X|×|Y| matrix representation of sampled point
    A: |X|×|X||Y| coefficient matrix of the equality constraint
    b: |X||Y|×1 right-hand side of the equality constraint
    P: |X||Y|×|X||Y| projection matrix onto the row space of A
    center: |X|×|Y| barycenter of the probability simplex
    rng: numpy's random state for reproducablity
    
    Outputs:
    new_pi: |X|×|Y| matrix representation of the new sampled point
    '''
    n, m = center.shape
    
    # Project to the affine subspace. We assume pi_initial already lives
    # there, but this will help with accumulation of numerical error.
    pi = project_onto_prob_simplex(pi, A, b, P, center).reshape(n*m,)

    # Choose a random direction.
    direction = rng.normal(size = n*m)

    # Project to subspace of admissible directions.
    direction = direction - np.matmul(P, direction)

    # Renormalize.
    direction /= np.linalg.norm(direction)

    # Determine how far to move while staying in the polytope — these
    # are inequality bounds, so we just need the entries to stay positive.
    is_pos = direction > 1e-6
    is_neg = direction < -1e-6
    
    direction_pos = direction[is_pos]
    direction_neg = direction[is_neg]
    
    pi_pos = pi[is_pos]
    pi_neg = pi[is_neg]
    
    # Choose a random distance to move.
    lower = np.max(-pi_pos / direction_pos)
    upper = np.min(-pi_neg / direction_neg)
    
    r = (upper - lower)*np.random.uniform() + lower

    # Calculate a new sample point.
    pi_new = (pi + r*direction).reshape(n, m)

    return pi_new

def sample_points(n, m, n_samples, n_skips, rng=None):
    '''
    Sample an ensemble of points from the polytope of row-stochastic
    matrices of size |X|×|Y| using MCMC.
    
    Inputs:
    n: |X|
    m: |Y|
    n_samples: number of points to sample
    n_skips: number of generated points to burn after each sampled point
    
    Outputs:
    points: list of |X|×|Y| matrix representations of the sampled points
    '''
    rng = rng or np.random.RandomState(0)

    A, b, center = create_prob_simplex(n, m)
    
    # Find orthonormal basis for row space of A.
    Q = linalg.orth(A.T)
    # Create projector onto the row space of A.
    P = np.matmul(Q, Q.T)

    n_steps = n_samples * n_skips

    points = []
    pi = center
    rng = np.random.RandomState(0)
    for i in range(n_steps):
        pi_new = markov_hit_and_run_step(pi, A, b, P, center, rng=rng)
        pi = pi_new
        if i % n_skips == 0:
            points.append(pi_new)

    return points
