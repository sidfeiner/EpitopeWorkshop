import functools
import random
from typing import Callable, Dict, Optional, Any, Set, List
import pandas as pd
import numpy as np


class UnknownValueToBalance(Exception):
    pass


VAL_COL_NAME = 'val'
MASK_COL_NAME = 'msk'


class Masker:
    def __init__(self):
        pass

    def update_mask(self, row: pd.Series, balance_val: Any, val_proba: float):
        if row[VAL_COL_NAME] != balance_val:
            return row[MASK_COL_NAME]
        return random.random() <= val_proba

    def get_balancer_mask(self, mask_df: pd.DataFrame, balance_val: Any, handled_vals: List[Any],
                          balances: Dict[Any, float]) -> pd.DataFrame:
        counts = mask_df[~mask_df[VAL_COL_NAME].isin(handled_vals)][VAL_COL_NAME].apply(
            lambda x: x == balance_val).value_counts(
            normalize=False, sort=True, ascending=True)
        minimum_freq_is_balance_val = counts.index[0]
        minimum_amt = counts.iloc[0]
        desired_freq = balances[balance_val] if minimum_freq_is_balance_val else sum(
            [balance for v, balance in balances.items() if v != balance_val and balance not in handled_vals])
        next_series_size = minimum_amt / desired_freq
        val_proba = next_series_size * (1-desired_freq) / counts[True]
        update_mask_func = functools.partial(self.update_mask, balance_val=balance_val, val_proba=val_proba)
        mask_df[MASK_COL_NAME] = mask_df.apply(update_mask_func, axis=1)
        return mask_df


class DataBalancer:
    """
    Rebalances data based on the values in some column
    """

    def __init__(self, balance_by_col: str, transform_val_func: Callable = id,
                 balances: Optional[Dict[Any, float]] = None):
        """
        :param balance_by_col: Column to balance by
        :param transform_val_func: If given, will transform the data in the target column and then 
                                   balance based on the transformed values
        :param balances: Dictionary of every value and the percentage of its values in the final
                                dataset. For example {'strong': 0.4, 'weak': 0.6} will make sure that in the
                                final dataset, 40% of the rows will be the records with 'strong' in `balance_by_col`
                                and 60% will be weak.
                                If not given, data will be completely balanced (equal representation of every value)
        """
        self.balance_by_col = balance_by_col
        self.transform_func = transform_val_func
        self.balances = balances

    def balance(self, df: pd.DataFrame):
        series = df[self.balance_by_col]
        transformed_vals = series.apply(self.transform_func)
        counts = transformed_vals.value_counts(normalize=True, sort=True, ascending=False)
        balances = self.balances if self.balances is not None else {value: 1 / len(counts) for value in counts.index}
        _balancer = Masker()
        handled_vals = []
        mask_df = pd.DataFrame(data={VAL_COL_NAME: transformed_vals})
        mask_df[MASK_COL_NAME] = True
        for val, freq in enumerate(counts):
            _balancer.get_balancer_mask(mask_df, val, handled_vals, balances)
            handled_vals.append(val)

        return df[mask_df[MASK_COL_NAME]]
