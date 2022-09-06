from functools import partial
from .base import BanditAlgo
from flax import struct
import jax.numpy as jnp
import jax

from jaxbandits.env import BanditEnvStep
@struct.dataclass
class UCB2State:
    step: int = 0
    counts: jnp.ndarray = None
    values: jnp.ndarray = None
    r: jnp.ndarray = None

    current_arm: int = 0
    current_arm_samples_left: int = 0

class UCB2(BanditAlgo):
    def __init__(self, arms, alpha = 0.5) -> None:
        super().__init__(arms)
        self.alpha = alpha
        self.step = 0

    @partial(jax.jit, static_argnames=["self"])
    def _tau(self, r: int):
        return jnp.ceil(jnp.power(1 + self.alpha, r)).astype(int)
    @partial(jax.jit, static_argnames=["self"])
    def _bonus(self, state: UCB2State) -> jnp.ndarray:
        rj = state.r
        a_n_rj = jnp.sqrt( 
            (1 + self.alpha) * jnp.log(
                jnp.e * state.step / self._tau(rj)
            ) / ( 2 * self._tau(rj))
        )
        return a_n_rj
    @partial(jax.jit, static_argnames=["self"])
    def sample(self, key, state: UCB2State) -> int:
        pass
    @partial(jax.jit, static_argnames=["self"])
    def reset(self) -> UCB2State:
        return UCB2State(
            step=0,
            counts=jnp.zeros((self.arms, ), dtype=int),
            values=jnp.zeros((self.arms, )),
            r=jnp.zeros((self.arms, ), dtype=int),
            current_arm=0,
            current_arm_samples_left=0
        )
    @partial(jax.jit, static_argnames=["self", "bandit_step_fn"])
    def update_step(self, key, state: UCB2State, bandit_state, bandit_step_fn: BanditEnvStep):
        key, sample_key, bandit_key = jax.random.split(key, 3)

        def sample_new_arm(state):
            bonus = self._bonus(state)
            a = jnp.argmax(state.values + bonus)
            return state.replace(
                current_arm=a,
                current_arm_samples_left=self._tau(state.r[a] + 1) - self._tau(state.r[a]),
                r=state.r.at[a].add(1)
            )
        state = jax.lax.cond(state.current_arm_samples_left == 0, sample_new_arm, lambda x : x, state)
        a = state.current_arm

        new_bandit_state, r = bandit_step_fn(bandit_key, bandit_state, a)

        n = state.counts[a]
        new_state = state.replace(
            step=state.step + 1,
            counts=state.counts.at[a].add(1),
            current_arm_samples_left=state.current_arm_samples_left - 1,
            values=state.values.at[a].set(
                (r + n * state.values[a]) / (n + 1)
            )
        )
        return new_state, new_bandit_state, a, r