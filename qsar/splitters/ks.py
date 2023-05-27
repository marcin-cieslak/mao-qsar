import numpy as np
import pandas as pd
import scipy

from qsar.data import split_data


def generate_seeds_ks(
    df: pd.DataFrame, split_method: str, n_seeds: int, search_size: int = 500
):
    """
    Use Kolmogorov-Smirnov D statistic to generate data split seeds that maximize the label
    distribution alignment between the training and testing subsets.

    Args:
        df: dataframe containing data (smiles and y columns are required).
        split_method: data splitting method (random or scaffold).
        n_seeds: number of seeds to generate
        search_size: number of seeds to test

    Returns:
        Random seeds that best align the label distribution.
    """
    seed_stats = []
    for seed in np.random.randint(10**6, size=search_size):
        np.random.seed(seed)
        train_idx, valid_idx, test_idx = split_data(
            df, method=split_method, return_idx=True
        )
        seed_stats.append(
            [
                seed,
                scipy.stats.ks_2samp(
                    df.iloc[test_idx].y,
                    df.iloc[train_idx].y,
                ).pvalue,
            ]
        )

    seed_stats = sorted(seed_stats, key=lambda x: -x[1])
    return [seed[0] for seed in seed_stats[:n_seeds]]
