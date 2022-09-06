from functools import partial
from .base import BanditAlgo
from flax import struct
import jax.numpy as jnp
import jax

from jaxbandits.envs import BanditEnvStep
@struct.dataclass
class UCB1State:
    step: int = 0
    # number of times we tried each action
    counts: jnp.ndarray = None
    values: jnp.ndarray = None

class UCB1(BanditAlgo):
    def __init__(self, arms) -> None:
        super().__init__(arms)
    @partial(jax.jit, static_argnames=["self"])
    def sample(self, key, state: UCB1State) -> int:
        a = jnp.argmax(state.values + jnp.sqrt( 2 * jnp.log(state.step) / state.counts))
        return a
    @partial(jax.jit, static_argnames=["self"])
    def reset(self) -> UCB1State:
        return UCB1State(
            step=0,
            counts=jnp.zeros((self.arms, )),
            values=jnp.zeros((self.arms, ))
        )
    @partial(jax.jit, static_argnames=["self", "bandit_step_fn"])
    def update_step(self, key, state: UCB1State, bandit_state, bandit_step_fn: BanditEnvStep):
        key, sample_key, bandit_key = jax.random.split(key, 3)
        a = self.sample(sample_key, state)

        new_bandit_state, r = bandit_step_fn(bandit_key, bandit_state, a)

        n = state.counts[a]
        new_state = state.replace(
            step=state.step + 1,
            counts=state.counts.at[a].add(1),
            values=state.values.at[a].set(
                (r + n * state.values[a]) / (n + 1)
            )
        )
        return new_state, new_bandit_state, a, r