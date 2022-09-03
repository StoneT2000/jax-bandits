from functools import partial

import jax
import jax.numpy as jnp
from jaxbandits.env import BernoulliBandits
from jaxbandits.algos.thompson import ThompsonSampling
import matplotlib.pyplot as plt
import numpy as np

@partial(jax.jit, static_argnames=["env_step", "update_step", "steps"])
def experiment(
    key: jax.random.KeyArray,
    env_state,
    algo_state,
    env_step,
    update_step,
    steps = 5000,
):
    cumulative_regret = []

    def body_fn(data, _):
        algo_state, env_state, key = data
        key, step_key = jax.random.split(key)
        algo_state, env_state, action, reward = update_step(step_key, algo_state, env_state, env_step)
        
        regret = jnp.max(env_state.arm_probs) - env_state.arm_probs[action]
        
        return (algo_state, env_state, key), dict(action=action, reward=reward, regret=regret)
    key, loop_key = jax.random.split(key, 2)

    _, agg = jax.lax.scan(body_fn, init = (algo_state, env_state, loop_key), xs=jnp.arange(0, steps))
    return agg


if __name__ == "__main__":

    
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    env = BernoulliBandits(arms=10)
    env_state = env.reset(key=reset_key)

    algo = ThompsonSampling(env.arms)
    algo_state = algo.reset()

    
    res = experiment(key, env_state, algo_state, env.step, algo.update_step, steps=500)
    # cumulative_regret = res["regret"]
    cumulative_regret = jnp.cumsum(res["regret"])
    
    
    # import ipdb;ipdb.set_trace()
    
    plt.plot(np.array(cumulative_regret))
    plt.show()