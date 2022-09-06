from functools import partial

from jaxbandits.envs.base import BanditEnv
from .base import BanditAlgo
from flax import struct
import jax.numpy as jnp
import jax

@struct.dataclass
class EpsilonGreedyState:
    step: int = 0
    # number of times we tried each action
    counts: jnp.ndarray = None
    values: jnp.ndarray = None

@struct.dataclass
class EpsilonGreedy(BanditAlgo):
    arms: int = struct.field(pytree_node=False)
    epsilon: int = struct.field(pytree_node=False)
    state: EpsilonGreedyState
    @classmethod
    def create(cls, arms, epsilon=0.01) -> "EpsilonGreedy":
        return cls(
            arms=arms,
            epsilon=epsilon,
            state=EpsilonGreedyState(
                step=0,
                counts=jnp.zeros((arms, )),
                values=jnp.zeros((arms, ))
            )
        )
    
    @jax.jit
    def sample(self, key) -> int:
        key, subkey = jax.random.split(key)
        a = jnp.where(jax.random.uniform(key) > self.epsilon, jnp.argmax(self.state.values), jax.random.randint(subkey, (), 0, self.arms))
        return a
    
    @jax.jit
    def update_step(self, key, env: BanditEnv):
        key, sample_key, bandit_key = jax.random.split(key, 3)
        a = self.sample(sample_key)

        env, r = env.step(bandit_key, a)

        n = self.state.counts[a]
        new_state = self.state.replace(
            step=self.state.step + 1,
            counts=self.state.counts.at[a].add(1),
            values=self.state.values.at[a].set(
                (r + n * self.state.values[a]) / (n + 1)
            )
        )
        return self.replace(state=new_state), env, a, r