from functools import partial

import jax
import jax.numpy as jnp

from jaxbandits.envs.base import BanditEnv
@partial(jax.jit, static_argnames=["env_step", "update_step", "steps"])
def experiment(
    key: jax.random.KeyArray,
    env: BanditEnv,
    algo_state,
    update_step,
    steps = 5000,
):
    def body_fn(data, _):
        algo_state, env, key = data
        env: BanditEnv
        key, step_key = jax.random.split(key)
        algo_state, env, action, reward = update_step(step_key, algo_state, env)
        regret = env.regret(action)
        
        return (algo_state, env, key), dict(action=action, reward=reward, regret=regret)
    key, loop_key = jax.random.split(key, 2)
    _, agg = jax.lax.scan(body_fn, init = (algo_state, env, loop_key), xs=jnp.arange(0, steps))
    return agg