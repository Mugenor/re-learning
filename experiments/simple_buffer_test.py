from myrl.memory.ReplayBuffer import ReplayBuffer
from myrl.spec.buffer import ValueSpec, ValueEnum, ValueStorageStrategyEnum
import numpy as np

if __name__ == '__main__':
    steps = 16
    n_workers = 2
    buffer = ReplayBuffer(capacity=16)
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
    data = buffer.sample(4)
    print(data.values)