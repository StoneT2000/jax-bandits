from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, TypeVar
import jax

EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvAction = TypeVar("EnvAction")
BanditEnvStep = Callable[[jax.random.KeyArray, EnvState, EnvAction], Tuple[EnvState, int]]
class BanditEnv(ABC):
    arms: int
    state: EnvState
    @classmethod
    def create(cls) -> "BanditEnv":
        pass
    @abstractmethod 
    def step(self, key, action):
        pass
    @abstractmethod
    def regret(self, action):
        pass