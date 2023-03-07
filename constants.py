SEED = 666

XY_GROUPS_PATH = '/home/me/mGH_UB/XY_groups'
# Metric space types.
ERDOS = 'Erdos–Renyi'
WATTS = 'Watts–Strogatz'
BARABASI = 'Barabasi-Albert'
POINT_CLOUD = 'point cloud'

# Step-size strategies for Frank-Wolfe.
CLOSED_LOOP = 'closed-loop' # constant γ
LINE_SEARCH = 'line-search' # linear-minimization γ

# Performance test types.
PERM_TEST = 'X🠖permute(X)'

# Performance dataframe columns.
VERSION_COL = 'version'
VERSION_DESCR_COL = 'descr'
TEST_TYPE_COL = 'test'
N_XYS_COL = 'n_spaces'
XY_SIZE_COL = 'space_size'
XY_TYPE_COL = 'space'
XY_PARAMS_COL = 'space_params'
N_RESTARTS_COL = 'n_restarts'
MAX_ITER_COL = 'max_iter'
C_SEQ_COL = 'c_seq'
CENTER_START_COL = 'center'
STEP_SIZE_COL = 'step_size'
AWAY_STEP_COL = 'away_step'

EXP_COLS = [XY_TYPE_COL, XY_SIZE_COL, XY_PARAMS_COL, N_RESTARTS_COL, MAX_ITER_COL, C_SEQ_COL,
            CENTER_START_COL, STEP_SIZE_COL, AWAY_STEP_COL, N_XYS_COL, TEST_TYPE_COL, VERSION_COL]
PERF_COLS = ['exact_mean', 'error_mean', 'rel_error_mean', 'time_mean', 'exact_std',
             'error_std', 'rel_error_std', 'time_std']
ALL_COLS_ORDERED = PERF_COLS[:4] + EXP_COLS + PERF_COLS[4:]

# Versions.
MIN_DIS_VERSIONS = {
    'v0': 'vanilla FW for c_exp',
    'v0.1': 'max_iter is shared by c_seq',
    'v0.1norm': 'scale X,Y by √‖X‖‖Y|'
}