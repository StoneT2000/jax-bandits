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

    # configure algorithm
    Algo = algos.UCB2
    Env = BernoulliBandits
    
    key, env_key = jax.random.split(key)
    env = Env.create(env_key, arms=16)

    algo = Algo.create(env.arms)


    stime = time.time()
    res = experiment(key, env, algo, N)
    compile_time = time.time() - stime
    print(f"Compile time: {compile_time:.6}s")

    stime = time.time()
    res = experiment(key, env, algo, N)
    elapsed_time = time.time() - stime
    print(f"Run time: {elapsed_time:.6}s")
    cumulative_regret = np.cumsum(np.array(res["regret"]))


    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.15, wspace=0.1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig.suptitle(f"{Algo.__name__} results on {Env.__name__}")
    ax1.set(xlabel="Samples", ylabel="Cumulative Regret")
    ax1.plot(np.array(cumulative_regret))
    ax2.set(xlabel="Samples", ylabel="Reward")
    ax2.plot(np.array(res["reward"]))
    plt.tight_layout()
    plt.show()
