import numpy as np
from scipy.stats import t


def confidence_error(df, confidence=0.95, stddev_col="stddev", n_col="n_loops"):
    alpha = 1.0 - confidence
    df_val = df[n_col] - 1
    t_multiplier = t.ppf(1 - alpha / 2, df=df_val)
    se = df[stddev_col] / np.sqrt(df[n_col])
    margin = t_multiplier * se
    return margin
