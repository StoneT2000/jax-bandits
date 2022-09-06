from abc import ABC, abstractmethod
from typing import Callable, Tuple, TypeVar
from jaxbandits.envs import BanditEnvStep, EnvState
from jaxbandits.envs.base import BanditEnv
AlgoState = TypeVar("AlgoState")

class BanditAlgo(ABC):
    arms: int
    @abstractmethod
    def update_step(self, key, env: BanditEnv) -> Tuple[AlgoState, BanditEnv, int, int]:
        """
        returns updated algo state, updated env state, selected action, and reward
        """
        pass
    # @abstractmethod
    # def reset(self) -> AlgoState:
    #     pass

# TODO, add a contextual bandits algo base