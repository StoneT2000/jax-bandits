from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from jaxbandits.envs.base import BanditEnvStep, BanditEnv, EnvState

@struct.dataclass
class GaussianBanditsState:
    values: jnp.ndarray = None

@struct.dataclass
class GaussianBandits(BanditEnv):
    arms: int = struct.field(pytree_node=False)
    state: GaussianBanditsState
    @classmethod
    def create(cls, key, arms=16, mu = 0.0, sigma = 1.0):
        values = jax.random.normal(key, (arms, ))
        values = values * sigma + mu
        return cls(
            arms=arms,
            state=GaussianBanditsState(
                values=values,
            )
        )
    @partial(jax.jit, static_argnames=["self"])
    def step(self, key: jax.random.KeyArray, action: int):
        return self, jax.random.normal(key, ()) + self.state.values[action]
        
