import pandas as pd


def nest(d: dict) -> dict:
    """
    Convert a flat dictionary with tuple key to a nested dictionary
    """
    result = {}
    for key, value in d.items():
        target = result
        for k in key[:-1]:
            target = target.setdefault(k, {})
        target[key[-1]] = value
    return result


def df_to_nested_dict(df_multi_indexed_col: pd.DataFrame) -> dict:
    """
    Convert a dataframe with multi-indexed columns to a nested dictionary
    """
    d = df_multi_indexed_col.to_dict('index')
    return {k: nest(v) for k, v in d.items()}
