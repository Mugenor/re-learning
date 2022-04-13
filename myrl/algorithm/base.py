from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple, Dict, Union

import numpy as np

from myrl.agent.action import Actions, ObservationInput
from myrl.agent.base import Agent
from myrl.condition.learn_condition import LearnCondition
from myrl.env.base import Environment
from myrl.memory.base import AbstractReplayBuffer
from myrl.spec.buffer import ValueSpec, ValueEnum, ValueStorageStrategyEnum
from myrl.stats.StepCounter import StepCounter
from myrl.utils.algorithm import check_all_mandatory_values_specified, evaluate_agent


class Algorithm(ABC):
    def __init__(self,
                 name: str,
                 agent: Agent,
                 env: Environment,
                 buffer: AbstractReplayBuffer,
                 start_learn_after_condition: LearnCondition,
                 learn_condition: LearnCondition,
                 learning_end_condition: LearnCondition):
        """
        TODO
        :param name: name of algorithm
        :param agent:
        :param buffer:
        """
        self.name = name
        self._agent = agent
        self._action_spec = env.action_spec
        self._observation_spec = env.observation_spec
        self._n_envs = env.n_envs
        self._buffer = buffer
        self._agent_buffer_values_spec = self._agent.get_buffer_value_specs()
        self._buffer_value_specs = self.get_buffer_value_specs() + self._agent_buffer_values_spec
        if not self._buffer.is_initialized:
            self._buffer.init(self._buffer_value_specs, self._n_envs)
        self._is_training = False
        self.step_counter = StepCounter(self._n_envs)
        self._should_start_learn = start_learn_after_condition
        self._should_learn_now = learn_condition
        self._should_stop_learning = learning_end_condition

    def random_action(self, n: int = 1) -> Actions:
        """
        :param n: Number of actions to return
        :return: Actions object
        """
        return self._agent.random_action(n)

    def act(self, observations: np.array, internal_state: Dict[str, Any]= None) -> Tuple[Actions, Dict[str, Any]]:
        """
        Acting according to the current policy
        :param observations: Array of shape: [N, *OBS], where N - number of observations, *OBS - dimensions of observations
        :param internal_state: Internal state of agent for specific observation. Useful for recurrent NNs
            Either None in case of agent has no internal state or sequence of length N of internal states.
        :return: Tuple of actions and internal states (or None in case of no internal state).
        """
        with evaluate_agent(self._agent):
            return self._agent.act(ObservationInput(observations), internal_state)

    def observe(self, values: Dict[Union[str, ValueEnum], Any]):
        """
        Dictionary of values to be processed by algorithm.
        Keys specify value name.
        Values for each algorithm/ are specified in get_buffer_value_specs method of algorithm and agent
        """
        check_all_mandatory_values_specified(values, self._buffer_value_specs)
        if self.is_training:
            dones = values.get(ValueEnum.DONE.name, None)
            self.step_counter.inc_dones(dones)
            self.step_counter.inc_steps()
            preprocessed_values = self._preprocess_values_before_adding_to_buffer(values)
            self._buffer.add(preprocessed_values)
            if self._should_start_learn(self.step_counter) and self._should_learn_now(self.step_counter):
                self.learn()

    def _preprocess_values_before_adding_to_buffer(self, values):
        return values

    @abstractmethod
    def learn(self):
        """
        Performing a learning algorithm by accumulated data
        """
        assert self.is_training
        self.step_counter.inc_learn()

    def get_buffer_value_specs(self) -> Sequence[ValueSpec]:
        """
        :return: List of metrics, to initialize memory buffer
        """
        return [
            ValueSpec(ValueEnum.OBSERVATION, self._observation_spec.shape),
            ValueSpec(ValueEnum.ACTION, self._action_spec.shape),
            ValueSpec(ValueEnum.REWARD, (1,)),
            ValueSpec(ValueEnum.DONE, (1,)),
            ValueSpec(ValueEnum.NEXT_OBSERVATION, self._observation_spec.shape,
                      storage_strategy=[ValueStorageStrategyEnum.LAST_PER_ENV]),
        ]

    @property
    def should_continue_learning(self):
        return not self._should_stop_learning(self.step_counter)

    @property
    def is_training(self) -> bool:
        return self._is_training

    def train(self):
        self._is_training = True
        self._agent.train()
        return self

    def eval(self):
        self._is_training = False
        self._agent.eval()
        return self
