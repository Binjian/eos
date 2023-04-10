from typing import Callable, List

import numpy as np


def nan_helper_1d(y: np.array) -> (np.array, Callable):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper_1d(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    sources: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_interp_1d(y: np.array) -> np.array:
    """Linear interpolation of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - y, 1d numpy array with NaNs replaced by linear interpolation
    Example:
        >>> # linear interpolation of NaNs
        >>> y = nan_interp_1d(y)

    sources: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    nans, x = nan_helper_1d(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def ragged_nparray_list_interp(ragged_list_list: List[List], ob_num: int) -> np.array:
    """Linear interpolation of NaNs.

    Input:
        - y, list ragged array
    Output:
        - y, 2d numpy array with missing elements replaced by linear interpolation
    Example:
        >>> # linear interpolation of NaNs
        >>> y = nan_interp_1d(y)

    sources: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.1f}'.format}, linewidth=100):
    with np.printoptions(suppress=True, linewidth=100):
        # capture warning about ragged json arrays
        with np.testing.suppress_warnings() as sup:
            log_warning = sup.record(
                np.VisibleDeprecationWarning,
                "Creating an ndarray from ragged nested sequences",
            )
            ragged_nparray_list = np.array(ragged_list_list)
            if len(log_warning) > 0:
                log_warning.pop()
                item_len = [len(item) for item in ragged_nparray_list]
                for count, item in enumerate(ragged_nparray_list):
                    if item_len[count] < ob_num:
                        ragged_nparray_list[count] = np.array(
                            list(ragged_nparray_list[count])
                            + [None] * (ob_num - item_len[count]),
                            dtype=np.float32,
                        )
                        ragged_nparray_list[count] = nan_interp_1d(
                            ragged_nparray_list[count]
                        )
                    elif item_len[count] > ob_num:
                        ragged_nparray_list[count] = ragged_nparray_list[count][:ob_num]
                    else:
                        pass
                aligned_nparray = np.array(list(ragged_nparray_list), dtype=np.float32)
            else:
                aligned_nparray = ragged_nparray_list
    return aligned_nparray
