from functools import partial

import jax
import jax.numpy as jnp
from jaxbandits import BernoulliBandits, experiment, algos
import matplotlib.pyplot as plt
import numpy as np
import time




if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    import numpy as np

    # Global flag to set a specific platform, must be used at startup.
    jax.config.update('jax_platform_name', 'cpu')

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    env = BernoulliBandits(arms=16)
    env_state = env.reset(key=reset_key)

    algo = algos.EpsilonGreedy(env.arms)
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