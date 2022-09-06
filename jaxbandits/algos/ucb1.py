from functools import partial
from .base import BanditAlgo
from flax import struct
import jax.numpy as jnp
import jax

from jaxbandits.env import BanditEnvStep
@struct.dataclass
class UCB1State:
    step: int = 0
    # number of times we tried each action
    counts: jnp.ndarray = None
    estimate_probs: jnp.ndarray = None

class UCB1(BanditAlgo):
    def __init__(self, arms, init_prob = 1.0) -> None:
        super().__init__(arms)
        self.init_prob = init_prob
        self.step = 0
    @partial(jax.jit, static_argnames=["self"])
    def sample(self, key, state: UCB1State) -> int:
        a = jnp.argmax(state.estimate_probs + jnp.sqrt( 2 * jnp.log(state.step) / state.counts))
        return a
    @partial(jax.jit, static_argnames=["self"])
    def reset(self) -> UCB1State:
        return UCB1State(
            step=0,
            counts=jnp.zeros((self.arms, )),
            estimate_probs=jnp.ones((self.arms, )) * self.init_prob
        )
    @partial(jax.jit, static_argnames=["self", "bandit_step_fn"])
    def update_step(self, key, state: UCB1State, bandit_state, bandit_step_fn: BanditEnvStep):
        key, sample_key, bandit_key = jax.random.split(key, 3)
        a = self.sample(sample_key, state)

        new_bandit_state, r = bandit_step_fn(bandit_key, bandit_state, a)

        new_state = state.replace(
            step=state.step + 1,
            counts=state.counts.at[a].add(1),
            estimate_probs=state.estimate_probs.at[a].add(

                (r - state.estimate_probs[a]) / (state.counts[a] + 1)
            )
        )
        return new_state, new_bandit_state, a, r