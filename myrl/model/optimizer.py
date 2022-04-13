from collections import Sequence
from typing import Optional, Callable, Any, Dict

import torch as T


class ConstantLR(T.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class Optimizer:
    def __init__(self,
                 base_optimizer: T.optim.Optimizer,
                 scheduler: Optional[T.optim.lr_scheduler._LRScheduler] = None,
                 pre_step_hooks: Optional[Sequence[Callable]] = None):
        self._base_optimizer = base_optimizer
        if scheduler is None:
            self._scheduler = ConstantLR(base_optimizer)
        else:
            self._scheduler = scheduler
        self._pre_step_hooks = [] if pre_step_hooks is None else pre_step_hooks

    def zero_grad(self):
        self._base_optimizer.zero_grad()

    def step(self):
        for hook in self._pre_step_hooks:
            hook()
        self._base_optimizer.step()
        self._scheduler.step()

    def optimize(self, loss: T.Tensor, **backward_params):
        self.zero_grad()
        loss.backward(**backward_params)
        self.step()

    def state_dict(self):
        return {'optimizer': self._base_optimizer.state_dict(),
                'scheduler': self._scheduler.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._base_optimizer.load_state_dict(state_dict['optimizer'])
        self._scheduler.load_state_dict(state_dict['scheduler'])
