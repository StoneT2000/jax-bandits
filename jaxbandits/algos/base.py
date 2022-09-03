from abc import ABC, abstractmethod
from typing import Callable, Tuple, TypeVar
from jaxbandits.env import BanditEnvStep, EnvState
AlgoState = TypeVar("AlgoState")

class BanditAlgo(ABC):
    def __init__(self, arms) -> None:
        self.arms = arms
    @abstractmethod
    def act(self, key, state: AlgoState) -> int:
        """
        choose the best action to maximize immediate return
        """
        pass
    @abstractmethod
    def update_step(self, key, state: AlgoState, bandit_state, bandit_step_fn: BanditEnvStep) -> Tuple[AlgoState, EnvState, int, int]:
        """
        returns updated algo state, updated env state, selected action, and reward
        """
        pass
    @abstractmethod
    def reset(self) -> AlgoState:
        pass

# TODO, add a contextual bandits algo base