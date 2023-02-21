import numpy as np
from itertools import permutations, product, combinations

from tools import *


def C(f, g, X, Y):
    delta = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            delta = max(delta, abs(X[i, g[j]] - Y[j, f[i]]))
        
    return delta


def dH_fX_Y(f, Y):
    is_fX = np.isin(np.arange(len(Y)), f)
    
    return 0 if is_fX.all() else np.max(np.min(Y[np.ix_(is_fX, ~is_fX)], axis=0))
    

def dH_fgY_Y(f, g, Y):
    is_fgY = np.isin(np.arange(len(Y)), np.array(f)[list(g)])
    
    return 0 if is_fgY.all() else np.max(np.min(Y[np.ix_(is_fgY, ~is_fgY)], axis=0))


def dH_fgY_fX(f, g, Y):
    is_fgY = np.isin(np.arange(len(Y)), np.array(f)[np.array(g)])
    is_fX_not_fgY = ~is_fgY & np.isin(np.arange(len(Y)), f)

    return np.max(np.min(Y[np.ix_(is_fgY, is_fX_not_fgY)], axis=0)) if is_fX_not_fgY.any() else 0


def dGH_fX_Y(f, Y):
    is_fX = np.isin(np.arange(len(Y)), f)
    
    return dGH(Y[np.ix_(is_fX, is_fX)], Y)[0]


def dGH_fgY_Y(f, g, Y):
    is_fgY = np.isin(np.arange(len(Y)), np.array(f)[list(g)])
    
    return dGH(Y[np.ix_(is_fgY, is_fgY)], Y)[0]


def dGH_fgY_fX(f, g, Y):
    is_fgY = np.isin(np.arange(len(Y)), np.array(f)[np.array(g)])
    is_fX = np.isin(np.arange(len(Y)), f)
    
    return dGH(Y[np.ix_(is_fgY, is_fgY)], Y[np.ix_(is_fX, is_fX)])[0]


def dmGH(X, Y):
    diam_X, diam_Y = (diam(D) for D in [X, Y])
    rad_X, rad_Y = (rad(D) for D in [X, Y])
    min_dis_lb = abs(diam_X - diam_Y)
    
    min_dis_f = min_dis_g = np.inf
    dis_all_f, dis_all_g = dict(), dict()

    for f in gen_fs(X, Y):
        min_dis_f = min(min_dis_f, dis(f, X, Y, dis_all_f))
        
    for g in gen_fs(Y, X):
        min_dis_g = min(min_dis_g, dis(g, Y, X, dis_all_g))
        
        if np.isclose(max(min_dis_f, min_dis_g), min_dis_lb):
            break
            
    return max(min_dis_f, min_dis_g)/2, dis_all_f, dis_all_g
    
    
def dGH(X, Y, finding_mGH=False, dis_all_f=None, dis_all_g=None):
    diam_X, diam_Y = (diam(D) for D in [X, Y])
    rad_X, rad_Y = (rad(D) for D in [X, Y])
    double_mGH_lb = abs(diam_X - diam_Y)
    double_GH_lb = max(double_mGH_lb, abs(rad_X - rad_Y))
    
    double_GH = double_mGH = np.inf
    dis_all_g = dis_all_g or dict()
        
    try:
        for f in gen_fs(X, Y):
            dis_f = dis(f, X, Y, dis_all_f)
            if not finding_mGH and dis_f > double_GH:
                continue
                
            for g in gen_fs(Y, X):
                dis_g = dis(g, Y, X, dis_all_g)
                if not finding_mGH and dis_g > double_GH:
                    continue

                max_dis = max(dis_f, dis_g)
                if max_dis < double_mGH:
                    double_mGH = max_dis
                    f_mGH = f
                    g_mGH = g

                if not np.isclose(double_GH, double_GH_lb):
                    max_dis_C = max(max_dis, C(f, g, X, Y))
                    if max_dis_C < double_GH:
                        double_GH = max_dis_C
                        f_GH = f
                        g_GH = g
                elif np.isclose(double_mGH, double_mGH_lb) or not finding_mGH:
                    raise StopIteration
    except StopIteration:
        pass
    
    results = [double_GH/2, f_GH, g_GH]
    if finding_mGH:
        results += [double_mGH/2, f_mGH, g_mGH]
    
    return results


def find_ratio(X, Y, verbose=False):
    GH, f_GH, g_GH, mGH, f_mGH, g_mGH = dGH(X, Y, finding_mGH=True)
    
    if verbose:
        print(f'mGH={mGH}: f={f_mGH}, g={g_mGH}')
        print(f'GH={GH}: f={f_GH}, g={g_GH}')

    return GH/mGH if mGH else 1


def find_ratio_fast(X, Y, verbose=False, ratio_wanted=1):
    mGH, dis_all_f, dis_all_g = dmGH(X, Y)
    if verbose:
        print(f'mGH={mGH}')

    if ratio_wanted > 1:
        for f in gen_fs(X, Y):
            if dis(f, X, Y, dis_all_f) < 2*mGH and dH_fX_Y(f, Y) < (ratio_wanted - 1)*mGH:
                return 0

        for g in gen_fs(Y, X):
            if dis(g, Y, X, dis_all_g) < 2*mGH and dH_fX_Y(g, X) < (ratio_wanted - 1)*mGH:
                return 0

            
    GH, f_GH, g_GH = dGH(X, Y, finding_mGH=False, dis_all_f=dis_all_f, dis_all_g=dis_all_g)

    if verbose:
        print(f'GH={GH}: f={f_GH}, g={g_GH}')

    return GH/mGH if mGH < GH else 1


def find_max_dGH_to_nested_subsets_ratio(X):
    max_ratio = 0
    n = len(X)
    for m in range(3, n):
        for ys in combinations(range(n), m):
            Y = X[np.ix_(ys, ys)] # Y ⊂ X
            dGH_YX = dGH(X, Y)[0]
            for k in range(2, m):
                for zs in combinations(ys, k):
                    Z = X[np.ix_(zs, zs)] # Z ⊂ Y

                    ratio = dGH_YX / dGH(X, Z)[0]
                    if ratio > max_ratio:
                        print(f'ratio={ratio} for X={repr(X)}, Y={repr(Y)}, Z={repr(Z)}')
                        max_ratio = ratio
