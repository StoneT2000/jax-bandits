from functools import partial

import jax
import jax.numpy as jnp
from jaxbandits.algos.base import BanditAlgo

from jaxbandits.envs.base import BanditEnv

@partial(jax.jit, static_argnames=["steps"])
def experiment(
    key: jax.random.KeyArray,
    env: BanditEnv,
    algo: BanditAlgo,
    steps = 5000,
):
    def body_fn(data, _):
        algo, env, key = data
        algo: BanditAlgo
        env: BanditEnv
        key, step_key = jax.random.split(key)
        algo, env, action, reward = algo.update_step(step_key, env)
        regret = env.regret(action)
        
        return (algo, env, key), dict(action=action, reward=reward, regret=regret)
    key, loop_key = jax.random.split(key, 2)
    _, agg = jax.lax.scan(body_fn, init = (algo, env, loop_key), xs=jnp.arange(0, steps))
    return agg