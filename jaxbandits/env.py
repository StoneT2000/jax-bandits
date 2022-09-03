from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Tuple, TypeVar
import jax
import jax.numpy as jnp
from flax import struct

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

@struct.dataclass
class BernoulliBanditsState:
    arm_probs: jnp.ndarray = None

class BernoulliBandits(BanditEnv):
    def __init__(self, arms: int) -> None:
        self.arms = arms

    @partial(jax.jit, static_argnames=["self"])
    def step(self, key: jax.random.KeyArray, state: BernoulliBanditsState, action: int):
        p = jax.random.uniform(key=key)
        return state, jnp.where(p < state.arm_probs[action], 1, 0)

    @partial(jax.jit, static_argnames=["self"])
    def reset(self, key: jax.random.KeyArray = None) -> EnvState:
        arm_probs = jax.random.uniform(key, (self.arms, ))
        return BernoulliBanditsState(arm_probs=arm_probs)
        
