from functools import partial
from typing import Callable

from jaxbandits.envs.base import BanditEnv
from .base import BanditAlgo
from flax import struct
import jax.numpy as jnp
import jax

from jaxbandits.envs import BanditEnvStep
@struct.dataclass
class ThompsonSamplingState:
    alphas: jnp.ndarray = None
    betas: jnp.ndarray = None

class ThompsonSampling(BanditAlgo):
    def __init__(self, arms, init_alpha = 1, init_beta = 1) -> None:
        super().__init__(arms)
        self.init_alpha = init_alpha
        self.init_beta = init_beta
    @partial(jax.jit, static_argnames=["self"])
    def sample(self, key, state: ThompsonSamplingState) -> int:
        ps = jax.random.beta(key=key, a=state.alphas, b=state.betas, shape=(self.arms, ))
        a = jnp.argmax(ps)
        return a
    @partial(jax.jit, static_argnames=["self"])
    def reset(self) -> ThompsonSamplingState:
        return ThompsonSamplingState(
            alphas=jnp.ones((self.arms, )),
            betas=jnp.ones((self.arms, ))
        )
    @partial(jax.jit, static_argnames=["self"])
    def update_step(self, key, state: ThompsonSamplingState, env: BanditEnv):
        key, sample_key, bandit_key = jax.random.split(key, 3)
        a = self.sample(sample_key, state)
        env, r = env.step(bandit_key, a)

        new_state = state.replace(
            alphas=state.alphas.at[a].add(r),
            betas=state.betas.at[a].add(1 - r)
        )
        return new_state, env, a, r