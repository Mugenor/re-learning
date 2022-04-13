from typing import Sequence, Union, Dict, Any

import numpy as np
import torch as T


def iterate_batches_indices(elements_length: int, batch_size: int):
    """
    Creates an iterator, which yields batched permuted indices from 0 to elements_length-1
    """
    assert 0 < batch_size <= elements_length
    # indices = np.random.permutation(elements_length)
    indices = np.arange(elements_length)
    for i in range(0, elements_length, batch_size):
        yield indices[i:i + batch_size]


def iterate_batches(sequences: Union[Dict[Any, Union[T.Tensor, np.ndarray]], Sequence[Union[T.Tensor, np.ndarray]]],
                    batch_size: int,
                    slice_over_dim: int = 0):
    """
    Creates an iterator, which yields batched permuted entries of sequences.
    :param sequences: contains sequences from which batches are created.
        The length of sequences must be equal. Also dict can be consumed
    :param batch_size: Batch size
    :param slice_over_dim: dimension index, which will be used for slicing
    """
    seq_len = None
    for seq in sequences.values() if isinstance(sequences, dict) else sequences:
        if seq_len is None:
            seq_len = seq.shape[slice_over_dim]
            assert batch_size <= seq_len
        else:
            assert seq_len == seq.shape[slice_over_dim]
    dim_slices = [slice(None) for i in range(slice_over_dim+1)]
    for indices in iterate_batches_indices(seq_len, batch_size):
        dim_slices[slice_over_dim] = indices
        if isinstance(sequences, dict):
            yield {key: seq[tuple(dim_slices)] for key, seq, in sequences.items()}
        else:
            yield (seq[tuple(dim_slices)] for seq in sequences)