from myrl.memory.recurrent.buffer import RecurrentReplayBuffer
from myrl.spec.buffer import ValueSpec, ValueEnum, ValueStorageStrategyEnum
import numpy as np

if __name__ == '__main__':
    steps = 16
    n_workers = 2
    buffer = RecurrentReplayBuffer(capacity=16, max_sequence_length=4)
    buffer.init([
        ValueSpec(ValueEnum.DONE, (1,)),
        ValueSpec(ValueEnum.ACTION, (1,)),
        ValueSpec(ValueEnum.OBSERVATION, (4,)),
        ValueSpec(ValueEnum.INTERNAL_STATE, (1,),
                  storage_strategy=[ValueStorageStrategyEnum.LAST_FROM_PREVIOUS_SEQUENCE],
                  default_value=np.zeros((1,)))
    ], n_envs=n_workers)
    for i in range(16):
        buffer.add({
            'DONE': np.array([not (i+1) % 6, not (i+1) % 4]).reshape(2, -1),
            'ACTION': np.arange(n_workers).reshape(n_workers, -1) + i*n_workers,
            'OBSERVATION': np.arange(n_workers * 4).reshape(n_workers, -1) + i*(n_workers*4),
            'INTERNAL_STATE': np.arange(n_workers * 1).reshape(n_workers, -1) + i*(n_workers*1)
        })
    data = buffer.all_data
    for batch, internal_state, lengths in data.iterate_batches(3):
        print(batch, internal_state, lengths)