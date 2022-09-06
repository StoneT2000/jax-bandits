from functools import partial

from jaxbandits.envs.base import BanditEnv
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


@struct.dataclass
class UCB1(BanditAlgo):
    arms: int = struct.field(pytree_node=False)
    state: UCB1State

    @classmethod
    def create(cls, arms):
        return cls(
            arms=arms,
            state=UCB1State(step=0,
                            counts=jnp.zeros((arms, )),
                            values=jnp.zeros((arms, )))
        )

    @jax.jit
    def update_step(self, key, env: BanditEnv):
        key, bandit_key = jax.random.split(key, 2)
        a = jnp.argmax(self.state.values + jnp.sqrt(2 *
                       jnp.log(self.state.step) / self.state.counts))

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
