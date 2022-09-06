from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, TypeVar
import jax

EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvAction = TypeVar("EnvAction")
BanditEnvStep = Callable[[jax.random.KeyArray, EnvState, EnvAction], Tuple[EnvState, int]]
class BanditEnv(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod 
    def step(self, action):
        pass
    @abstractmethod
    def reset(self, key):
        pass