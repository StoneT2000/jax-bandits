import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxbandits import BernoulliBandits, algos, experiment

if __name__ == "__main__":

    # Global flag to set a specific platform, must be used at startup.
    jax.config.update("jax_platform_name", "cpu")

    # configure starting seed and max samples
    key = jax.random.PRNGKey(0)
    N = 4096
    
    key, env_key = jax.random.split(key)
    env = BernoulliBandits.create(env_key, arms=16)

    algo = algos.ThompsonSampling.create(env.arms)


    stime = time.time()
    res = experiment(key, env, algo, N)
    compile_time = time.time() - stime
    print(f"Compile time: {compile_time}s")

    stime = time.time()
    res = experiment(key, env, algo, N)
    elapsed_time = time.time() - stime
    print(f"Run time: {elapsed_time}s")
    cumulative_regret = np.cumsum(np.array(res["regret"]))

    fig, axs = plt.subplots(1, 2)

    axs[0].set(xlabel="Samples", ylabel="Cumulative Regret")
    axs[0].plot(np.array(cumulative_regret))
    axs[0].set(xlabel="Samples", ylabel="Reward")
    axs[1].plot(np.array(res["reward"]))
    plt.show()
