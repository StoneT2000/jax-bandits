from functools import partial

import jax
import jax.numpy as jnp
@partial(jax.jit, static_argnames=["env_step", "update_step", "steps"])
def experiment(
    key: jax.random.KeyArray,
    env_state,
    algo_state,
    env_step,
    update_step,
    steps = 5000,
):
    def body_fn(data, _):
        algo_state, env_state, key = data
        key, step_key = jax.random.split(key)
        algo_state, env_state, action, reward = update_step(step_key, algo_state, env_state, env_step)
        
        regret = jnp.max(env_state.arm_probs) - env_state.arm_probs[action]
        
        return (algo_state, env_state, key), dict(action=action, reward=reward, regret=regret)
    key, loop_key = jax.random.split(key, 2)
    _, agg = jax.lax.scan(body_fn, init = (algo_state, env_state, loop_key), xs=jnp.arange(0, steps))
    return agg