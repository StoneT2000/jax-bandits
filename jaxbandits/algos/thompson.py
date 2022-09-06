from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from jaxbandits.envs import BanditEnvStep
from jaxbandits.envs.base import BanditEnv

from .base import BanditAlgo


@struct.dataclass
class ThompsonSamplingState:
    alphas: jnp.ndarray = None
    betas: jnp.ndarray = None


@struct.dataclass
class ThompsonSampling(BanditAlgo):
    arms: int = struct.field(pytree_node=False)
    state: ThompsonSamplingState

    @classmethod
    def create(cls, arms) -> "ThompsonSampling":
        return cls(
            arms=arms,
            state=ThompsonSamplingState(
                alphas=jnp.ones((arms,)), betas=jnp.ones((arms,))
            ),
        )

    @jax.jit
    def sample(self, key: jax.random.KeyArray) -> int:
        ps = jax.random.beta(
            key=key, a=self.state.alphas, b=self.state.betas, shape=(self.arms,)
        )
        a = jnp.argmax(ps)
        return a

    @jax.jit
    def update_step(self, key, env: BanditEnv):
        key, sample_key, bandit_key = jax.random.split(key, 3)
        a = self.sample(sample_key)
        env, r = env.step(bandit_key, a)

        new_state = self.state.replace(
            alphas=self.state.alphas.at[a].add(r),
            betas=self.state.betas.at[a].add(1 - r),
        )
        return self.replace(state=new_state), env, a, r
