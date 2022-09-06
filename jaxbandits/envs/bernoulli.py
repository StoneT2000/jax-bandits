from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from jaxbandits.envs.base import BanditEnvStep, BanditEnv, EnvState

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
        
