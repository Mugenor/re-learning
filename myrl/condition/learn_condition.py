from abc import abstractmethod

from myrl.condition.base import Condition
from myrl.stats.StepCounter import StepCounter


class LearnCondition(Condition):
    @abstractmethod
    def __call__(self, step_counter: StepCounter) -> bool:
        raise NotImplementedError


class EverySteps(LearnCondition):
    def __init__(self, every_steps: int):
        self.every_steps = every_steps

    def __call__(self, step_counter: StepCounter) -> bool:
        return (step_counter.steps + 1) % self.every_steps == 0


class EveryTotalSteps(LearnCondition):
    def __init__(self, every_total_steps: int):
        self.every_total_steps = every_total_steps

    def __call__(self, step_counter: StepCounter) -> bool:
        return (step_counter.total_steps + 1) % self.every_total_steps == 0


class EveryLearn(LearnCondition):
    def __init__(self, every_learns_done: int):
        self.every_learns_done = every_learns_done

    def __call__(self, step_counter: StepCounter) -> bool:
        return (step_counter.learn_times + 1) % self.every_learns_done == 0


class AfterLearnsDone(LearnCondition):
    def __init__(self, learns_done: int):
        self.learns_done = learns_done

    def __call__(self, step_counter: StepCounter) -> bool:
        return step_counter.learn_times > self.learns_done


class AfterEpisodesCompleted(LearnCondition):
    def __init__(self, episodes_completed: int):
        self.episodes_completed = episodes_completed

    def __call__(self, step_counter: StepCounter):
        return step_counter.dones.sum() >= self.episodes_completed


class AfterStepsDone(LearnCondition):
    def __init__(self, steps_done: int):
        self.steps_done = steps_done

    def __call__(self, step_counter: StepCounter):
        return step_counter.steps > self.steps_done