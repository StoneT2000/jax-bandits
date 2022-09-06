from functools import partial

import jax
import jax.numpy as jnp
from jaxbandits.algos.ucb1 import UCB1
from jaxbandits.algos.ucb2 import UCB2
from jaxbandits.env import BernoulliBandits
from jaxbandits.algos.thompson import ThompsonSampling
import matplotlib.pyplot as plt
import numpy as np
import time

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


if __name__ == "__main__":

    import jax
    import jax.numpy as jnp

    # Global flag to set a specific platform, must be used at startup.
    jax.config.update('jax_platform_name', 'cpu')
    
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    env = BernoulliBandits(arms=16)
    env_state = env.reset(key=reset_key)

    algo = UCB2(env.arms)
    algo_state = algo.reset()

    N = 4096
    stime = time.time()
    res = experiment(key, env_state, algo_state, env.step, algo.update_step, steps=N)
    compile_time = time.time() - stime
    print(f"Compile time: {compile_time}s")
    
    stime = time.time()
    res = experiment(key, env_state, algo_state, env.step, algo.update_step, steps=N)
    elapsed_time = time.time() - stime
    print(f"Run time: {elapsed_time}s")
    cumulative_regret = np.cumsum(np.array(res["regret"]))
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Samples")
    plt.plot(np.array(cumulative_regret))
    plt.show()