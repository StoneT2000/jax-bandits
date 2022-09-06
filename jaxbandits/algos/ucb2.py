from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from jaxbandits.envs.base import BanditEnv

from .base import BanditAlgo


@struct.dataclass
class UCB2State:
    step: int = 0
    counts: jnp.ndarray = None
    values: jnp.ndarray = None
    r: jnp.ndarray = None

    current_arm: int = 0
    current_arm_samples_left: int = 0


@struct.dataclass
class UCB2(BanditAlgo):
    alpha: int = struct.field(pytree_node=False)
    arms: int = struct.field(pytree_node=False)
    state: UCB2State

    @classmethod
    def create(cls, arms, alpha=0.5):
        return cls(
            alpha=alpha,
            arms=arms,
            state=UCB2State(
                step=0,
                counts=jnp.zeros((arms,), dtype=int),
                values=jnp.zeros((arms,)),
                r=jnp.zeros((arms,), dtype=int),
                current_arm=0,
                current_arm_samples_left=0,
            ),
        )

    @jax.jit
    def _tau(self, r: int):
        return jnp.ceil(jnp.power(1 + self.alpha, r)).astype(int)

    @jax.jit
    def _bonus(self, state: UCB2State) -> jnp.ndarray:
        rj = state.r
        a_n_rj = jnp.sqrt(
            (1 + self.alpha)
            * jnp.log(jnp.e * state.step / self._tau(rj))
            / (2 * self._tau(rj))
        )
        return a_n_rj

    @jax.jit
    def update_step(self, key, env: BanditEnv):
        key, bandit_key = jax.random.split(key)

        def sample_new_arm(state: UCB2State):
            bonus = self._bonus(state)
            a = jnp.argmax(state.values + bonus)
            return state.replace(
                current_arm=a,
                current_arm_samples_left=self._tau(state.r[a] + 1)
                - self._tau(state.r[a]),
                r=state.r.at[a].add(1),
            )

        new_state = jax.lax.cond(
            self.state.current_arm_samples_left == 0,
            sample_new_arm,
            lambda x: x,
            self.state,
        )
        new_state: UCB2State
        a = new_state.current_arm

        env, r = env.step(bandit_key, a)

        n = new_state.counts[a]
        new_state = new_state.replace(
            step=new_state.step + 1,
            counts=new_state.counts.at[a].add(1),
            current_arm_samples_left=new_state.current_arm_samples_left - 1,
            values=new_state.values.at[a].set((r + n * new_state.values[a]) / (n + 1)),
        )
        return self.replace(state=new_state), env, a, r
