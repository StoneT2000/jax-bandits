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

@struct.dataclass
class ThompsonSampling(BanditAlgo):
    arms: int = struct.field(pytree_node=False)
    state: ThompsonSamplingState
    @classmethod
    def create(cls, arms) -> "ThompsonSampling":
        # self.init_alpha = init_alpha
        # self.init_beta = init_beta
        return cls(
            arms=arms,
            state=ThompsonSamplingState(
            alphas=jnp.ones((arms, )),
            betas=jnp.ones((arms, ))
            )
        )
    
    # @partial(jax.jit, static_argnames=["self"])
    @jax.jit
    def sample(self, key: jax.random.KeyArray) -> int:
        ps = jax.random.beta(key=key, a=self.state.alphas, b=self.state.betas, shape=(self.arms, ))
        a = jnp.argmax(ps)
        return a
    # @partial(jax.jit, static_argnames=["self"])
    # def reset(self) -> ThompsonSamplingState:
    #     return ThompsonSamplingState(
    #         alphas=jnp.ones((self.arms, )),
    #         betas=jnp.ones((self.arms, ))
    #     )
    # @partial(jax.jit, static_argnames=["self"])
    @jax.jit
    def update_step(self, key, env: BanditEnv):
        key, sample_key, bandit_key = jax.random.split(key, 3)
        a = self.sample(sample_key)
        env, r = env.step(bandit_key, a)

        new_state = self.state.replace(
            alphas=self.state.alphas.at[a].add(r),
            betas=self.state.betas.at[a].add(1 - r)
        )
        return self.replace(state=new_state), env, a, r