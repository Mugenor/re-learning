from types import GeneratorType
from typing import Union, Sequence, Dict, Any, Iterable, Tuple

import numpy as np
import torch as T

from myrl.spec.env import ActionSpec


def np_to_tensor(input: Union[np.ndarray, Sequence[np.array]],
                 device: T.device = None,
                 dtype: T.dtype = None,
                 share_memory: bool = False) -> Union[np.array, Sequence[np.array]]:
    """
    Converts numpy array(s) to pytorch tensor(s) and puts it on device
    :param input: numpy array or iterable or dictionary
    :param device: pytorch device which will be used to allocate tensors. None for default tensor type
    :param dtype: dtype of new tensors
    :param share_memory: In case if device is cpu, it is possible to reuse the underlying memory.
        Set this flog to True if you want this behaviour
    :return: Pytorch tensor(s) on specified device
    """
    assert (share_memory and device.type == 'cpu' and dtype is None) or not share_memory
    if not isinstance(input, (np.ndarray, dict)) and isinstance(input, Iterable):
        if share_memory:
            generator = (np_to_tensor(array, device, dtype, share_memory)
                         for array in input)
        else:
            generator = (np_to_tensor(array, device, dtype, share_memory)
                         for array in input)
        if isinstance(input, GeneratorType):
            return generator
        else:
            return input.__class__(generator)
    elif isinstance(input, dict):
        return {key: np_to_tensor(array, device, dtype, share_memory)
                for key, array in input.items()}
    else:
        if share_memory:
            return T.from_numpy(input) if not isinstance(input, T.Tensor) else input
        else:
            return T.tensor(input, dtype=dtype, device=device) if not isinstance(input, T.Tensor) else input


def unpack_dict(dictionary: Dict, keys: Sequence[Any]):
    remaining_keys = dictionary.keys() - keys
    unpacked_values = tuple(dictionary[key] for key in keys)
    remaining_values = {key: dictionary[key] for key in remaining_keys}
    return unpacked_values, remaining_values


def concatenate_sequences(sequences: Union[np.ndarray, Sequence[np.ndarray]],
                          lengths: np.ndarray) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Concatenates padded sequences
    :param sequences: Of shape [N, L, *]
    :param lengths: Sequence lengths of shape [N]
    :return: Concatenated sequences of shape [sum(lengths), *]
    """
    is_one_ndarray = isinstance(sequences, np.ndarray)
    if is_one_ndarray:
        is_all_sequences_full = lengths.sum() == (sequences.shape[0] * sequences.shape[1])
        flatten_sequences = sequences.reshape((-1, *sequences.shape[2:]))
        L_dim = sequences.shape[1]
    else:
        is_all_sequences_full = lengths.sum() == (sequences[0].shape[0] * sequences[0].shape[1])
        flatten_sequences = [sequences_entry.reshape((-1, *sequences_entry.shape[2:])) for sequences_entry in sequences]
        L_dim = sequences[0].shape[1]
    if is_all_sequences_full:
        return flatten_sequences
    else:
        sequence_indices = tuple(np.arange(length) + i * L_dim for i, length in enumerate(lengths))
        sequence_indices = np.concatenate(sequence_indices, axis=0)
        if is_one_ndarray:
            return flatten_sequences[sequence_indices]
        else:
            return [flatten_sequences_entry[sequence_indices] for flatten_sequences_entry in flatten_sequences]


def np_pad_axis(array: np.ndarray, pad_width: Tuple[int, int], axis: int) -> np.ndarray:
    if pad_width == (0, 0):
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = pad_width
    return np.pad(array, npad)


def split_to_sequences(array: np.ndarray,
                       lengths: np.ndarray) -> np.ndarray:
    """
    Splits data in array to padded sequences of length lengths
    :param array: of shape [N, *]
    :param lengths: of shape [l], where l - is the number of sequences
    :return: array of shape [l, L, *], here L is the maximum length of sequence
    """
    assert lengths.sum() == array.shape[0]
    sequences_list = np.split(array, indices_or_sections=np.cumsum(lengths[:-1]), axis=0)
    max_length = len(max(sequences_list, key=len))
    padded_sequences = [np_pad_axis(sequence, (0, max_length - len(sequence)), axis=0) for sequence in sequences_list]
    return np.stack(padded_sequences)


def index_in_dict(dictionary: Dict, indices: Any):
    return {key: value[indices] for key, value in dictionary.items()}


def create_sequence_mask(lengths: Union[np.ndarray, T.Tensor]) -> Union[np.ndarray, T.Tensor]:
    """
    Returns boolean mask for sequences
    :param lengths: 1D array of sequence lengths
    :return: 2D mask of booleans [N, L], where N - number of sequences, L - max length.
    """
    max_length = lengths.max()
    seq_count = len(lengths)
    if isinstance(lengths, np.ndarray):
        return np.tile(np.arange(max_length), (seq_count, 1)) < lengths.reshape(seq_count, 1)
    else:
        return T.tile(T.arange(max_length), (seq_count, 1)) < lengths.reshape(seq_count, 1)


def scale_continuous_action(action: Union[np.ndarray, T.Tensor],
                            action_spec: ActionSpec) -> Union[np.ndarray, T.Tensor]:
    """
    Transforms actions from [-1:1] to [low:high]
    """
    low, high = action_spec.min, action_spec.max
    return low + (action + 1.0) * (high-low) / 2.0


def unscale_continuous_action(action: Union[np.ndarray, T.Tensor],
                              action_spec: ActionSpec) -> Union[np.ndarray, T.Tensor]:
    """
    Transforms actions from [low:high] to [-1:1]
    """
    low, high = action_spec.min, action_spec.max
    if isinstance(action, T.Tensor):
        low, high = T.from_numpy(low), T.from_numpy(high)
    return 2.0 * ((action-low) / (high-low)) - 1.0
