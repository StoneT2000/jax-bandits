from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from flax import struct
from jaxbandits.envs.base import BanditEnvStep, BanditEnv, EnvState
import flax
@struct.dataclass
class BernoulliBanditsState:
    arm_probs: jnp.ndarray = None
    optimal_action: int = None

@struct.dataclass
class BernoulliBandits(BanditEnv):

    arms: int
    state: BernoulliBanditsState

    @classmethod
    def create(cls, key: jax.random.KeyArray, arms: int):
        arm_probs = jax.random.uniform(key, (arms, ))
        return cls(
            arms=arms,
            state=BernoulliBanditsState(
                arm_probs=arm_probs,
                optimal_action=jnp.argmax(arm_probs)
            )
        )

    @jax.jit
    def step(self, key: jax.random.KeyArray, action: int):
        p = jax.random.uniform(key=key)
        return self, jnp.where(p < self.state.arm_probs[action], 1, 0)
    
    @jax.jit
    def regret(self, action: int) -> float:
        return self.state.arm_probs[self.state.optimal_action] - self.state.arm_probs[action]
