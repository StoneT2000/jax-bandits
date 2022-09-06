from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from jaxbandits.envs.base import BanditEnvStep, BanditEnv, EnvState

@struct.dataclass
class GaussianBanditsState:
    arm_probs: jnp.ndarray = None

class GaussianBandits(BanditEnv):
    def __init__(self, arms: int, mu: float = 0, sigma: float = 1) -> None:
        self.arms = arms
        self.mu = mu
        self.sigma = sigma

    @partial(jax.jit, static_argnames=["self"])
    def step(self, key: jax.random.KeyArray, state: GaussianBanditsState, action: int):
        p = jax.random.uniform(key=key)
        return state, state.arm_probs[action]

    @partial(jax.jit, static_argnames=["self"])
    def reset(self, key: jax.random.KeyArray = None) -> EnvState:
        arm_probs = jax.random.normal(key, (self.arms, ))
        arm_probs = arm_probs * self.sigma + self.mu
        return GaussianBanditsState(arm_probs=arm_probs)
        
