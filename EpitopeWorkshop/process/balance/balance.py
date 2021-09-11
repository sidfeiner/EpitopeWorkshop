import functools
import random
from typing import Callable, Dict, Optional, Any, List, Tuple
import pandas as pd


class UnknownValueToBalance(Exception):
    pass


VAL_COL_NAME = 'val'
MASK_COL_NAME = 'msk'


def self(x):
    return x


SELF_FUNC = self


class UnderSamplingMasker:

    @classmethod
    def update_mask(cls, row: pd.Series, balance_val: Any, val_proba: float, *args, **kwargs):
        if row[VAL_COL_NAME] != balance_val:
            return row[MASK_COL_NAME]
        return random.random() <= val_proba

    @classmethod
    def update_balancer_mask(cls, mask_df: pd.DataFrame, balance_val: Any, handled_vals: List[Any],
                             balances: Dict[Any, float], cur_final_series_size: int) -> int:
        """
        Inplace, updates the mask of the df
        Return final series size based on this value's rebalance
        """
        counts = mask_df[~mask_df[VAL_COL_NAME].isin(handled_vals)][VAL_COL_NAME].apply(
            lambda x: x == balance_val).value_counts(
            normalize=False, sort=True, ascending=True)
        minimum_freq_is_balance_val = counts.index[0]
        minimum_amt = counts.iloc[0]
        desired_freq = balances[balance_val] if minimum_freq_is_balance_val else sum(
            [balance for v, balance in balances.items() if v != balance_val and balance not in handled_vals])
        next_series_size = min(cur_final_series_size, minimum_amt / desired_freq)
        val_proba = next_series_size * desired_freq / counts[True]
        update_mask_func = functools.partial(cls.update_mask, balance_val=balance_val, val_proba=val_proba)
        mask_df[MASK_COL_NAME] = mask_df.apply(update_mask_func, axis=1)
        return next_series_size


class OverSamplingDefiner:
    """
    This class will update a column to set the amount of time a record should be duplicated
    """

    class RemaningDuplicates:
        def __init__(self, amt_org_rows: int, total_rows_to_duplicate: int):
            self.org_rows_amt = amt_org_rows
            self.total_rows_to_duplicate = total_rows_to_duplicate
            self.remaning_rows_to_duplicate = total_rows_to_duplicate

    @classmethod
    def update_mask(cls, row: pd.Series, balance_val: Any, duplicates: int, *args, **kwargs):
        if row[VAL_COL_NAME] != balance_val:
            return row[MASK_COL_NAME]
        return duplicates

    @classmethod
    def update_balancer_col(cls, mask_df: pd.DataFrame, balance_val: Any, handled_vals: List[Any],
                            balances: Dict[Any, float], cur_final_series_size: int) -> int:
        counts = mask_df[~mask_df[VAL_COL_NAME].isin(handled_vals)][VAL_COL_NAME].apply(
            lambda x: x == balance_val).value_counts(
            normalize=False, sort=True, ascending=False)
        maximum_freq_is_balance_val = counts.index[0]
        maximum_amt = counts.iloc[0]
        desired_freq = balances[balance_val] if maximum_freq_is_balance_val else sum(
            [balance for v, balance in balances.items() if v != balance_val and balance not in handled_vals])
        next_series_size = max(cur_final_series_size, maximum_amt / desired_freq)
        records_to_duplicate = next_series_size * desired_freq - counts[True]
        duplicates_per_row = int(records_to_duplicate // counts[True])
        update_mask_func = functools.partial(cls.update_mask, balance_val=balance_val, duplicates=duplicates_per_row)
        mask_df[MASK_COL_NAME] = mask_df.apply(update_mask_func, axis=1)
        return next_series_size


class DataBalancer:
    """
    Rebalances data based on the values in some column
    """

    def __init__(self, balance_by_col: str, transform_val_func: Callable = SELF_FUNC,
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
        raise NotImplementedError()


class UnderSamplingBalancer(DataBalancer):
    """
    Rebalances data based on the values in some column using undersampling
    """

    def __init__(self, balance_by_col: str, transform_val_func: Callable = SELF_FUNC,
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
        super().__init__(balance_by_col, transform_val_func, balances)

    def balance(self, df: pd.DataFrame):
        series = df[self.balance_by_col]
        transformed_vals = series.apply(self.transform_func)
        counts = transformed_vals.value_counts(normalize=True, sort=True, ascending=True)
        balances = self.balances if self.balances is not None else {value: 1 / len(counts) for value in counts.index}
        handled_vals = []
        mask_df = pd.DataFrame(data={VAL_COL_NAME: transformed_vals})
        mask_df[MASK_COL_NAME] = True
        series_size = len(df)
        for val, freq in counts.items():
            series_size = UnderSamplingMasker.update_balancer_mask(mask_df, val, handled_vals, balances, series_size)
            handled_vals.append(val)

        return df[mask_df[MASK_COL_NAME]]


class OverSamplingBalancer(DataBalancer):
    """
    Rebalances data based on the values in some column using undersampling
    """

    def __init__(self, balance_by_col: str, col_to_transform: List[Tuple[str, Callable]],
                 transform_val_func: Callable = SELF_FUNC,
                 balances: Optional[Dict[Any, float]] = None):
        """
        :param balance_by_col: Column to balance by
        :param col_to_transform: Every column name and the transformation that should be applied to it
        :param transform_val_func: If given, will transform the data in the target column and then
                                   balance based on the transformed values
        :param balances: Dictionary of every value and the percentage of its values in the final
                                dataset. For example {'strong': 0.4, 'weak': 0.6} will make sure that in the
                                final dataset, 40% of the rows will be the records with 'strong' in `balance_by_col`
                                and 60% will be weak.
                                If not given, data will be completely balanced (equal representation of every value)
        """
        super().__init__(balance_by_col, transform_val_func, balances)
        self.col_to_transform = col_to_transform

    def set_duplicates_amt_col(self, df: pd.DataFrame):
        series = df[self.balance_by_col]
        transformed_vals = series.apply(self.transform_func)
        df[VAL_COL_NAME] = transformed_vals
        counts = transformed_vals.value_counts(normalize=True, sort=True, ascending=False)
        balances = self.balances if self.balances is not None else {value: 1 / len(counts) for value in counts.index}
        handled_vals = []
        series_size = len(df)
        for val, freq in enumerate(counts):
            series_size = OverSamplingDefiner.update_balancer_col(df, val, handled_vals, balances, series_size)
            handled_vals.append(val)
        df.drop(VAL_COL_NAME, axis=1, inplace=True)
        return df

    def duplicate_row(self, row: pd.Series):
        duplicates = row[MASK_COL_NAME]
        if duplicates == 0:
            return {col: [row[col]] for col, _ in self.col_to_transform}
        return {col: [row[col]] + [transform(row[col]) for _ in range(duplicates)] for col, transform in
                self.col_to_transform}

    def prepare_duplicated_data(self, df: pd.DataFrame):
        expanded_cols = df.apply(self.duplicate_row, axis=1, result_type='expand')
        for original_col in list(expanded_cols.columns):
            df.drop(original_col, axis=1, inplace=True)
        return pd.concat([df, expanded_cols], axis=1)

    def add_duplicates(self, df: pd.DataFrame):
        df = self.prepare_duplicated_data(df)
        columns = [col for (col, _) in self.col_to_transform]
        return df.explode(columns, ignore_index=True)

    def balance(self, df: pd.DataFrame):
        df[MASK_COL_NAME] = 0
        df_with_duplicates_amt = self.set_duplicates_amt_col(df)
        df = self.add_duplicates(df_with_duplicates_amt)
        df.drop(MASK_COL_NAME, axis=1, inplace=True)
        return df
