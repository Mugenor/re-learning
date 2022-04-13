from typing import Union, Tuple

import numpy as np
import torch as T


def assert_correct_shape(array: Union[np.array, T.Tensor], expected_shape: Tuple[int, ...]):
    """
    Checks correctness of array shape
    :param array: Numpy array or pytorch tensor
    :param expected_shape: Tuple which describes the shape.
        If next value is -1, then check is not performed.
        If length of shape is shorter then length of array.shape, then only len(shape) first dimensions are checked
    """
    for dim, expected_dim in zip(array.shape, expected_shape):
        if expected_dim >= 0:
            assert dim == expected_dim, f'Wrong shape: {array.shape} <-> {expected_shape}'
