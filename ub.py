import numpy as np
from functools import partial
import scipy.linalg as la
from scipy.optimize import minimize, linear_sum_assignment, LinearConstraint, Bounds

from constants import SEED
from tools import dis, rnd_P, central_P, f_to_F, P_to_f, project_P, card



def find_descent_direction(grad_at_P, injective=False):
    """
    Finds F ∈ [0,1]^|X|×|Y| that minimizes cosine similarity
    (dot product) with the gradient at some P ∈ [0,1]^|X|×|Y|;
    F will always be in {0,1}^|X|×|Y|.

    :param grad_at_P: gradient at P, |X|×|Y| matrix
    :param injective: whether to search on injective mappings only, bool
    :return: F, |X|×|Y| matrix
    """
    if injective:  # injectivity assumes |X| ≤ |Y|
        f = linear_sum_assignment(grad_at_P)[1]
    else:
        f = np.argmin(grad_at_P, axis=1)
        #rows, cols = np.where(grad_value == grad_value.min(axis=1)[:, None])#!!to randomize argmin
        #f = np.array([np.random.choice(cols[rows == i]) for i in range(len(grad_value))])#!!

    return f_to_F(f, grad_at_P.shape[1])


def c_exp_hess(c, X, Y):
    return np.kron(c ** X, c ** -Y) + np.kron(c ** -X, c ** Y)


def conv(H): # H is Hessian
    """
    Quantify convexity of the objective function as the ratio of total
    magnitude of positive eigenvalues of its Hessian to the total magnitude
    of all its eigenvalues.

    :param H: Hessian of the objective
    :return: a number in [0, 1]
    """
    eigs = la.eigvalsh(H)

    return np.sum(eigs[eigs > 0]) / np.sum(np.abs(eigs))


# Frank-Wolfe algoritm.
def solve_frank_wolfe(X, Y, u, grad_u, L, grad_u_wrt_gamma, dis_XY, P0, tol=1e-8, max_iter=10,
                      verbose=0, injective=False, step_size='line-search', check_away_step=False):
    """
    Minimize u:R^|X|×|Y|🠖R over the mapping polytope 𝒫 ⊂ [0,1]|X|×|Y|.

    :param X: X or |X|
    :param Y: Y or |Y|
    :param u: objective
    :param grad_u: ∇u
    :param L: Lipschitz constant of ∇u
    :param grad_u_wrt_gamma:
    :param dis_XY: function dis:(X🠖Y)🠖R
    :param P0: starting point
    :param tol:
    :param max_iter: maximum number of iterations
    :param verbose: how much information to print, 0–2
    :param injective: whether to constraint 𝒫 to injective mappings only, bool
    :param step_size: 'line-search' or 'closed-loop, see
        https://hal.science/hal-03579585/document
    :param check_away_step: whether to check stepping away from traversed
        vertices for better descent, bool
    :return: solution, number of iterations
    """
    P = P0.copy()

    # Init decomposition of P by traversed vertices in ℱ (descent directions).
    if check_away_step:
        # Init sequence of the traversed vertices so that P can be decomposed
        # as their linear combination (+ starting point).
        F_seq = np.zeros((max_iter + 1, card(X), card(Y)))
        F_seq[0] = P
        # Init sequence of coefficients of the decomposition of P.
        alpha_seq = np.zeros(max_iter + 1)
        alpha_seq[0] = 1

    # On each iteration of Frank-Wolfe algorithm...
    for i in range(1, max_iter + 1):
        grad_at_P = grad_u(P)
        gamma = None

        # Find the default Frank-Wolfe direction.
        F = find_descent_direction(grad_at_P, injective=injective)
        D = F - P
        direction_point, direction_point_desc = F, '(to) F'
        gap = np.sum(-grad_at_P * D) # rate of decrease in chosen direction
        if step_size == 'line-search':
            gamma_max = 1
        elif step_size == 'closed-loop':
            assert L is not None, 'Lipschitz constant for ∇u is not given'
            gamma = min(gap / L / la.norm(D, 2), 1)
        else:
            raise ValueError(f'unknown step-size strategy {step_size}')

        if check_away_step:
            # Find the away direction.
            F_dot_grad_seq = np.sum(F_seq * grad_at_P, axis=(1, 2))
            F_dot_grad_seq[np.isclose(alpha_seq, 0)] = -np.inf
            F_idx = np.argmax(F_dot_grad_seq)
            D_away = P - F_seq[F_idx]
            gap_away = np.sum(-grad_at_P * D_away)

            # Choose the away direction if it has a steeper decrease in u
            # than the default Frank-Wolfe direction.
            if gap_away > gap:
                D = D_away
                direction_point, direction_point_desc = F_seq[F_idx], '(from) S'
                gap = gap_away
                gamma_max = alpha_seq[F_idx] / (1 - alpha_seq[F_idx])

                alpha_seq_increase_sign = 1
            else:
                # Update P decomposition with newly traversed vertex otherwise.
                F_seq[i] = F
                F_idx = i
                alpha_seq_increase_sign = -1

        if verbose > 0:
            # Describe the direction point as a mapping vector.
            F = project_P(P)
            if verbose > 1:
                direction_point_desc += f'={P_to_f(direction_point)}'
                proj_P_desc = f', proj P={P_to_f(F)}'
            else:
                proj_P_desc = ''

        # Find how much to move in the decided direction.
        if gamma is None:
            # res = min((minimize(#!! only start from gamma_max/2 and ensure no difference from gamma0 in [0, gamma_max]
            #     lambda x: u(P + x * direction), gamma0, bounds=[(0, gamma_max)], jac=lambda x: alpha_jac(x, P, direction))
            #     for gamma0 in [0, gamma_max]), key=lambda res: res.fun)
            res = minimize(
                lambda x: u(P + x * D), gamma_max/2, bounds=[(0, gamma_max)],
                jac=lambda x: grad_u_wrt_gamma(x, P, D))
            gamma = res.x[0]

        # Calculate the step from P.
        P_increase = gamma * D

        # Check if the rate of decrease is too small to proceed.
        if gap < tol:
            if verbose > 0:
                print('-' * 10 + f'STOP: {direction_point_desc}, γ={gamma:.5f}, '
                                 f'‖∇u(P)‖²={la.norm(grad_at_P, 2):.2f}, '
                                 f'‖γD‖²={la.norm(P_increase, 2):.2f}',
                      '' if res.success else '(FAILURE)')
            break

        assert np.allclose(np.sum(P + P_increase, axis=1), 1), \
            f'next P is not row-stochastic: γ={gamma}, D={D}, prev P={P}'

        if verbose > 0:# or not res.success:
            print(f'iter {i}: dis(P)={dis_XY(P):.3f}, dis(Proj P)={dis_XY(F)}, u(P)={u(P):.2f}'
                  f'{proj_P_desc}, {direction_point_desc}, γ={gamma:.5f}', '' if res.success else '(FAILURE)')

        # Take the step from P.
        P += P_increase

        # Update decomposition of P based on traversed vertices of 𝒫.
        if check_away_step:
            alpha_seq += alpha_seq_increase_sign * gamma
            alpha_seq[F_idx] -= alpha_seq_increase_sign * gamma

            assert np.all(alpha_seq >= -1e10),\
                f'traversed vertex weights are negative: {alpha_seq[alpha_seq < 0]}'
            assert np.allclose(P, np.sum(F_seq * alpha_seq[:, None, None], axis=0)),\
                f'bad decomposition of P by traversed vertices'

    if verbose > 0:
        print('-' * 10 + f'FINAL dis(P)={dis_XY(P):.3f}, dis(Proj P)={dis_XY(F)}, '
                         f'u(P)={u(P):.2f}{proj_P_desc}')
        if check_away_step:
            print(alpha_seq[alpha_seq > 0])

    return P, i - 1


def solve_frank_wolfe_seq(fw_seq, P0, **kwargs):
    """
    Solve a sequence of minimization problems using each solution as
    a subsequent starting point.

    :param fw_seq: Frank-Wolfe algorithms for solving each minimization
    :param P0: starting point for the first minimization in the sequence
    :param kwargs: common parameters of the Frank-Wolfe algorithms
    :return: solution, number of iterations in each problem
    """
    if type(fw_seq) is not list:
        fw_seq = [fw_seq]

    P = P0
    iterations = []
    for fw in fw_seq:
        P, i = fw(P0=P, **kwargs)
        iterations.append(i)

    return P, iterations


def make_frank_wolfe_solver(X, Y, c=2., u_type='c_exp', **kwargs):
    """
    Create Frank-Wolfe solver for minimizing c-based u:R^|X|×|Y|🠖R over the
    mapping polytope 𝒫 ⊂ [0,1]|X|×|Y|.

    :param X:
    :param Y:
    :param u_type:
    :param c:
    :param kwargs:
    :return: function for solving the minimization
    """
    # Make objective u.
    if u_type == 'sq':  # '‖Δ‖²'
        def S(P):
            return -X @ P @ Y
    elif u_type == 'sq+':  # '‖Δ‖² for non-surj'
        def S(P):
            return -X @ P @ Y + .5 * P @ Y @ P.T @ P @ Y
    elif u_type in {'exp', 'mix', 'mix+'}:  # '‖e^|Δ|‖_1'
        def S(P):
            return np.e**X @ P @ np.e**-Y + np.e**-X @ P @ np.e**Y
    elif u_type == 'sq+exp':  # '‖Δ‖² for non-surj'
        def S(P):
            return c**X @ P @ c**-Y + c**-X @ P @ c**Y - X @ P @ Y + .5 * P @ Y @ P.T @ P @ Y
    elif u_type == 'c_exp':  # '‖c^|Δ|‖_1'
        #delta = .1  # smallest positive |d_X(x, x') - d_Y(y, y')|
        #c = (len(X) ** 2 + 1) ** (1 / delta)  # so that argmin ‖c^|Δ|‖_1 = argmin ‖Δ‖_∞

        def S(P):
            return c**X @ P @ c**-Y + c**-X @ P @ c**Y
    elif u_type not in {'XP-PY', 'exp_2'}:
        raise ValueError(f'unknown objective type {u_type}')

    def u(P):
        if u_type == 'XP-PY':  # '‖XP-PY‖²'
            return np.sum((X @ P - P @ Y)**2)
        elif u_type == 'exp_2':
            return np.sum((P * (np.e**X @ P @ np.e**-Y + np.e**-X @ P @ np.e**Y))**2)
        else:
            return np.sum(P * S(P))

    # Make ∇u.
    def grad_u(P):
        if u_type == 'XP-PY':  # '‖XP-PY‖²'
            return 2*(X @ X @ P - 2*X @ P @ Y + P @ Y @ Y)
        elif u_type == 'sq+':
            return 2 * (-X @ P @ Y + P @ Y @ P.T @ P @ Y)
        elif u_type == 'sq+exp':
            return 2 * (c**X @ P @ c**-Y + c**-X @ P @ c**Y - X @ P @ Y + P @ Y @ P.T @ P @ Y)
        elif u_type == 'exp_2':
            T_0 = np.exp(X)
            T_1 = np.exp(-Y)
            T_2 = np.exp(-X)
            T_3 = np.exp(Y)
            T_4 = (((T_0).dot(P)).dot(T_1) + ((T_2).dot(P)).dot(T_3))
            T_5 = (P * T_4)
            T_6 = (T_5 * P)
            return ((2 * (T_5 * T_4)) + (2 * ((T_0.T).dot(T_6)).dot(T_1.T))) + (2 * ((T_2.T).dot(T_6)).dot(T_3.T))
        else:
            return 2 * S(P)

    # Calculate Lipschitz constant of ∇u.
    if u_type == 'c_exp':
        L = 2*(la.norm(c**(X - Y), 2) + la.norm(c**(Y - X), 2))
    else:
        L = None

    def grad_u_wrt_gamma(gamma, P, D):  # D = F - P
        if u_type == 'XP-PY':  # '‖XP-PY‖²'
            return 2 * (gamma * np.sum((X @ D - D @ Y) ** 2) + np.sum((X @ D) * (X @ P)) +
                        np.sum((D @ Y) * (P @ Y)) - np.sum((X @ D) * (P @ Y)) - np.sum((X @ P) * (D @ Y)))
        elif u_type == 'sq+':
            # !! MORE EFFICIENT FORMULATION (BUT YIELDS ERROR WHEN CALLING 'minimize()'
            # P_plus_alpha_D = P + alpha * d
            # return 2*(np.sum(P_plus_alpha_D * (d @ Y @ P_plus_alpha_D.T @ P_plus_alpha_D @ Y)) -\
            #           np.sum(P_plus_alpha_D * (X @ d @ Y)))

            PD = P.T @ D
            YPPY = Y @ P.T @ P @ Y
            YPDY = Y @ (PD + PD.T) @ Y
            YDDY = Y @ D.T @ D @ Y
            return 2 * gamma ** 3 * np.sum(D * (D @ YDDY)) + \
                   1.5 * gamma ** 2 * (np.sum(D * (P @ YDDY)) + np.sum(P * (D @ YDDY)) + np.sum(D * (D @ YPDY))) + \
                   gamma * (np.sum(P * (P @ YDDY)) + np.sum(D * (P @ YPDY)) + np.sum(P * (D @ YPDY)) + np.sum(
                D * (D @ YPPY))) + \
                   .5 * (np.sum(P * (P @ YPDY)) + np.sum(D * (P @ YPPY)) + np.sum(P * (D @ YPPY))) + \
                   np.sum(D * (-X @ P @ Y)) + np.sum(P * (-X @ D @ Y)) + 2 * gamma * np.sum(D * (-X @ D @ Y))
        # !! make grad_u_wrt_gamma for 'sq+exp' to try it out (sq+ should serve as a regularization)!!
        elif u_type == 'exp_2':
            T_0 = (P + (gamma * D))
            T_1 = np.exp(X)
            T_2 = np.exp(-Y)
            T_3 = np.exp(-X)
            T_4 = np.exp(Y)
            T_5 = ((T_1).dot((T_0).dot(T_2)) + (T_3).dot((T_0).dot(T_4)))
            T_6 = (T_0 * T_5)
            T_7 = (T_6 * T_0)
            return np.array([((2 * np.trace((D).dot((T_6 * T_5).T))) + (
                        2 * np.trace((((T_7.T).dot(T_1)).dot(D)).dot(T_2)))) + (
                                    2 * np.trace((((T_7.T).dot(T_3)).dot(D)).dot(T_4)))])

        else:
            return np.sum(D * S(P)) + np.sum(P * S(D)) + 2 * gamma * np.sum(D * S(D))

    # def alpha_hess(alpha, P, d):  # d is dQP = Q - P
    #     if obj_type == 'XP-PY':  # '‖XP-PY‖²'
    #         return 2 * np.sum((X @ d - d @ Y)**2)
    #     elif obj_type == 'sq+':
    #         PD = P.T @ d
    #         YPPY = Y @ P.T @ P @ Y
    #         YPDY = Y @ (PD + PD.T) @ Y
    #         YDDY = Y @ d.T @ d @ Y
    #         return 6 * alpha ** 2 * np.sum(d * (d @ YDDY)) + \
    #                3 * alpha * (np.sum(d * (P @ YDDY)) + np.sum(P * (d @ YDDY)) + np.sum(d * (d @ YPDY))) + \
    #                np.sum(P * (P @ YDDY)) + np.sum(d * (P @ YPDY)) + np.sum(P * (d @ YPDY)) + np.sum(d * (d @ YPPY)) + \
    #                2 * np.sum(d * (-X @ d @ Y))
    #     else:
    #         return 2 * np.sum(d * S(d))

    dis_XY = partial(dis, X=X, Y=Y)
    fw = partial(solve_frank_wolfe, len(X), len(Y), u, grad_u, L,
                 grad_u_wrt_gamma, dis_XY, **kwargs)

    return fw


def find_min_dis(X, Y, c_seq=2., n_restarts=1, center_start=True, max_iter=10, verbose=0,
                 injective=False, rnd=None):
    """
    Find minimum dis(f) for f:X🠖Y.

    :param X: X
    :param Y: Y
    :param c_seq:
    :param n_restarts: how many starting points to try, int
    :param center_start: whether to try the center of 𝒫 as a starting point first, bool
    :param max_iter: maximum number of Frank-Wolfe iterations, int
    :param injective: whether to consider only injective mappings, bool
    :param verbose:
    :return:
    """
    if type(c_seq) is not list:
        c_seq = [c_seq]

    n, m = card(X), card(Y)
    if injective:
        assert n <= m, '|X| cannot be greater than |Y| to allow for injectivity'

    # Initialize the starting points.
    rnd = rnd or np.random.RandomState(SEED)
    P0s = [rnd_P(n, m, rnd) for _ in range(n_restarts)]
    if center_start:
        P0s[0] = central_P(n, m)

    # Calculate maximum number of Frank-Wolfe iterations.
    if max_iter == 'log':
        max_iter = int(np.ceil(np.log(max(n, m))))

    fw_seq = [make_frank_wolfe_solver(X, Y, u_type='c_exp', c=c)
              for c in c_seq]

    min_dis = np.inf
    for P0 in P0s:
        P, _ = solve_frank_wolfe_seq(fw_seq, P0, max_iter=max_iter,
                                  injective=injective, verbose=verbose)
        f = P_to_f(P)
        dis_f = dis(f, X, Y)

        if dis_f < min_dis:
            best_f = f
            min_dis = dis_f

    return min_dis, best_f


# def find_ub_FAQ(X, Y, DX, DY, obj_type, max_iter, inj, verbose):
#     if len(X) > len(Y):  # ensure |X| ≤ |Y|
#         X, Y = Y, X
#         DX, DY = DY, DX
#
#     obj_XY, grad_XY, alpha_jac_XY = def_functions_for_frank_wolfe(X, Y, obj_type)
#     f_YX, grad_YX, alpha_jac_YX = def_functions_for_frank_wolfe(Y, X, obj_type)
#     dis_XY = partial(dis, DX=DX, DY=DY)
#     dis_YX = partial(dis, DX=DY, DY=DX)
#     frank_wolfe_XY = partial(
#         solve_frank_wolfe, n=len(X), m=len(Y), f=obj_XY, grad=grad_XY, dis_=dis_XY,
#         alpha_jac=alpha_jac_XY, max_iter=max_iter, verbose=verbose)
#     frank_wolfe_YX = partial(
#         solve_frank_wolfe, n=len(Y), m=len(X), f=f_YX, grad=grad_YX, dis_=dis_YX,
#         alpha_jac=alpha_jac_YX, max_iter=max_iter, verbose=verbose)
#
#     if inj == -1:
#         P_XY = frank_wolfe_XY(is_inj=False)
#         P_YX = frank_wolfe_YX(is_inj=False)
#         ub = max(dis_XY(project_P(P_XY)), dis_YX(project_P(P_YX))) / 2
#         inj_ub = np.nan
#     else:
#         if obj_type == 'mix':
#             inj_P = solve_frank_wolfe(len(X), len(Y), *def_functions_for_frank_wolfe(X, Y, 'sq'),
#                                       injective=True, dis_XY=dis_XY, max_iter=max_iter, verbose=verbose)
#         else:
#             inj_P = frank_wolfe_XY(is_inj=True)
#
#         inj_ub = dis_XY(project_P(inj_P)) / 2
#         print('inj ub is ', inj_ub)
#         if inj == 0:
#             P_XY = frank_wolfe_XY(P0=inj_P, is_inj=False)
#             P_YX = frank_wolfe_YX(P0=inj_P, is_inj=False)
#             ub = max(dis_XY(project_P(P_XY)), dis_YX(project_P(P_YX))) / 2
#         elif inj == 1:
#             ub = np.nan
#             P_XY = P_YX = inj_P
#
#     return inj_ub, ub, P_XY, P_YX

# def find_double_mGH(X, Y, double_lb=0, n_restarts=1, P0=None, max_iter=10, rng=None):
#
#     if not isinstance(P0, list):
#         P0 = [P0] * n_restarts
#     else:
#         n_restarts = len(P0)
#
#     rng = rng or np.random.RandomState(0)
#
#     dis_ = partial(dis, DX=X, DY=Y)
#     fw_seq = [make_frank_wolfe(X, Y, c=c, dis_=dis_, rng=rng) for c in c_seq]
#
#     min_dis = np.inf
#     for starting_point in P0:
#         P_star, _ = solve_frank_wolfe_seq(fw_seq, verbose=False, max_iter=max_iter, P0=starting_point)
#         min_dis = min(min_dis, dis_(project_P(P_star)))
#         if min_dis <= double_lb:
#             break
#
#     return min_dis
