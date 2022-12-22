# helper functions for various statistical analysis tests
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from funcsforprajay.funcs import flattenOnce


def tukey_hsd(grouped_data: dict, alpha=0.05):
    """Tukey's honestly significant difference (HSD) test for multiple comparisons of means."""

    # do some checking on the input data
    # check that the number of groups is at least 2
    if len([*grouped_data]) < 2:
        raise ValueError('At least 2 groups are required for Tukey HSD')

    print(f'Performing Tukey HSD test for {len([*grouped_data])} groups...')

    # create DataFrame to hold data
    data_nums = []
    for key, values in grouped_data.items():
        data_nums.extend([key] * len(values))

    df = pd.DataFrame({'score': flattenOnce(list(grouped_data.values())),
                       'group': data_nums})

    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=alpha)
    print(tukey)




