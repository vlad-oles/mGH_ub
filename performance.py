import os
import sys
import time
from itertools import product, combinations
from functools import partial
import numpy as np
import pandas as pd

from constants import *
from tools import *
from ub import find_min_dis


PERF_PATH = 'performance.csv'
MIN_DIS_VERSION = 'v0.1norm'
MIN_DIS_VERSION_DESCR = MIN_DIS_VERSIONS[MIN_DIS_VERSION]

# Init data params.
TEST_TYPE = [PERM_TEST]
N_XYS = [100]
POINT_CLOUD_METRIC = ['euclidean', 'cityblock']
ERDOS_P = [.05, .2]
WATTS_K = [1, 4]
WATTS_P = [.01, .05]
BARABASI_M = [1, 4]
XY_SIZE = [10, 100]

# Init algorithm params.
STEP_SIZE = [LINE_SEARCH] #[LINE_SEARCH, CLOSED_LOOP]
AWAY_STEP = [False] #[True, False]
C_SEQ = [[1.01, 1.1, 1.5, 2, 3], [1.1, 2], [2]]
N_RESTARTS = [1, 10]
MAX_ITER = [10, 100]
CENTER_START = [True, False]

# Parse algorithm params.
for arg in sys.argv[1:]:
    key, values = arg.split('=')
    if key == 'c':
        C_SEQ = [[float(c) for c in v.split(',')] for v in values.split(';')]
    if key == 'n':
        N_RESTARTS = [int(v) for v in values.split(';')]
    if key == 'i':
        MAX_ITER = [int(v) for v in values.split(';')]

# Set up metric spaces.
start = time.time()
XY_GROUPS = dict()
for test_type, n_XYs, XY_size in product(TEST_TYPE, N_XYS, XY_SIZE):

    # Set up metric space parameter sets.
    XY_param_sets = \
        [f'metric={metric}' for metric in POINT_CLOUD_METRIC] + \
        [f'p={p}' for p in ERDOS_P] + [f'k={k},p={p}' for k, p in product(WATTS_K, WATTS_P)] + \
        [f'm={m}' for m in BARABASI_M]

    # Set up metric space type for each set of metric space parameters
    XY_types = [POINT_CLOUD] * len(POINT_CLOUD_METRIC) + [ERDOS] * len(ERDOS_P) + \
               [WATTS] * len(WATTS_K) * len(WATTS_P) + [BARABASI] * len(BARABASI_M)

    # Set up metric space-generating function for each (metric space type, parameter set).
    gen_X_partials = \
        [partial(gen_point_cloud, metric=metric) for metric in POINT_CLOUD_METRIC] + \
        [partial(gen_graph, graph_type=ERDOS, p=erdos_p, enforce_n=True) for erdos_p in ERDOS_P] + \
        [partial(gen_graph, graph_type=WATTS, k=watts_k, p=watts_p, enforce_n=True)
         for watts_k, watts_p in product(WATTS_K, WATTS_P)] + \
        [partial(gen_graph, graph_type=BARABASI, m=barabasi_m, enforce_n=True)
         for barabasi_m in BARABASI_M]

    # Set up metric spaces for each (metric space type, parameter set).
    for gen_X_partial, X_params, X_type in zip(gen_X_partials, XY_param_sets, XY_types):

        XY_group_path = os.path.join(XY_GROUPS_PATH, f'{test_type},{n_XYs},{X_type},{X_params},{XY_size}.npz')

        try:
            # Read metric spaces.
            npz = np.load(XY_group_path)
        except FileNotFoundError:
            # Generate metric spaces.
            rnd = np.random.RandomState(SEED)

            if test_type == PERM_TEST:
                Xs = [gen_X_partial(XY_size, rnd=rnd) for _ in range(n_XYs)]
                Ys = [permute(X, rnd) for X in Xs]

                target_distortions = [0] * n_XYs
            else:
                raise ValueError(f'Unknown performance test type: {test_type}')

            # Store metric spaces.
            np.savez(XY_group_path, Xs=Xs, Ys=Ys, target_distortions=target_distortions)

        else:
            Xs, Ys, target_distortions = npz['Xs'], npz['Ys'], npz['target_distortions']

        XY_GROUPS[test_type, n_XYs, X_type, X_params, XY_size] = Xs, Ys, target_distortions

end = time.time()
print(f'{time.ctime()[:-5]} | Read (or generated) metric spaces in {(end - start)/60:.1f}m')

# Parse existing benchmarking results.
perf_by_exp = dict()
try:
    perf_df = pd.read_csv(PERF_PATH)
except FileNotFoundError:
    pass
else:
    for _, row in perf_df.iterrows():
        exp = tuple(row[EXP_COLS].values)
        perf = tuple(row[PERF_COLS].values)
        perf_by_exp[exp] = perf

# Benchmark for each metric space group.
start = time.time()
for (test_type, n_XYs, XY_type, XY_params, XY_size), (Xs, Ys, target_distortions) in XY_GROUPS.items():

    XY_group_start = time.time()

    n_skipped = n_experiments = 0
    for c_seq, n_restarts, center_start, max_iter, step_size, away_step in product(
            C_SEQ, N_RESTARTS, CENTER_START, MAX_ITER, STEP_SIZE, AWAY_STEP):

        c_seq_str = ','.join(str(c) for c in c_seq)
        exp = (XY_type, XY_size, XY_params, n_restarts, max_iter, c_seq_str, center_start, step_size,
               away_step, n_XYs, test_type, MIN_DIS_VERSION)

        # Skip experiment if its conditions are not compatible.
        if (step_size == CLOSED_LOOP and away_step) or (center_start and n_restarts > 1):
            continue

        n_experiments += 1

        # Skip experiment if present in the existing benchmarking results.
        if exp in perf_by_exp:
            n_skipped += 1
            continue

        # Measure performance of finding minimum distortion for all metric space pairs.
        are_exact = []
        errors = []
        times = []
        for X, Y, target_dis in zip(Xs, Ys, target_distortions):

            Y /= la.norm(X, 2) # v0.1norm
            X /= la.norm(X, 2) # v0.1norm

            exp_start = time.time()
            rnd = np.random.RandomState(SEED)
            min_dis, _ = find_min_dis(
                X, Y, c_seq=c_seq, n_restarts=n_restarts, center_start=center_start,
                max_iter=max_iter, step_size=step_size, check_away_step=away_step, rnd=rnd)
            exp_end = time.time()

            are_exact.append(int(min_dis == target_dis))
            errors.append(min_dis - target_dis)
            times.append(exp_end - exp_start)

        # Calculate performance metrics.
        errors = np.array(errors)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = errors / np.array(target_dis)
        perf = pd.DataFrame([are_exact, errors, rel_errors, times]).T.agg(
            ['mean', 'std']).values.flatten()

        perf_by_exp[exp] = perf

        # Store performance metrics.
        perf_df = pd.DataFrame(
            [[*exp, MIN_DIS_VERSION_DESCR, *perf] for exp, perf in perf_by_exp.items()],
            columns=EXP_COLS + [VERSION_DESCR_COL] + PERF_COLS)
        perf_df.sort_values(
            [VERSION_COL, TEST_TYPE_COL, N_XYS_COL, XY_SIZE_COL, XY_TYPE_COL, XY_PARAMS_COL,
             N_RESTARTS_COL, MAX_ITER_COL, C_SEQ_COL, CENTER_START_COL, STEP_SIZE_COL, AWAY_STEP_COL],
            ascending=[False, True, False, False, True, True, False, False, True, True, True, False],
            inplace=True)
        perf_df[ALL_COLS_ORDERED].to_csv(PERF_PATH, index=False)

    XY_group_end = time.time()
    print(f'{time.ctime()[:-5]} | [{n_experiments-n_skipped}/{n_experiments}] '
          f'{(XY_group_end - XY_group_start) / 60:.1f}m for {test_type} test '
          f'on {n_XYs} {XY_type} ({XY_params}) of size {XY_size}')


end = time.time()
print(f'{time.ctime()[:-5]} | {(end - start)/60:.1f}m for full performance test')


