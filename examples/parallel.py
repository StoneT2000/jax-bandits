from functools import partial

import jax
import jax.numpy as jnp
from jaxbandits import BernoulliBandits, experiment, algos
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial

import jax
import jax.numpy as jnp


if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    import numpy as np

    key = jax.random.PRNGKey(0)
    num_envs = 1024
    arms = 16
    N = 4096

    key, *env_keys = jax.random.split(key, num_envs + 1)
    env_batch = jax.vmap(BernoulliBandits.create, in_axes=[0, None])(jnp.stack(env_keys), arms)
    
    algo_batch = jax.vmap(algos.ThompsonSampling.create, in_axes=[None], axis_size=num_envs)(arms)

    experiment_vmapped = jax.vmap(experiment, in_axes=[0, 0, 0, None], axis_size=num_envs)

    stime = time.time()
    key, *exp_keys = jax.random.split(key, num_envs + 1)    
    res = experiment_vmapped(jnp.stack(exp_keys), env_batch, algo_batch, N)
    compile_time = time.time() - stime
    print(f"Compile time: {compile_time}s")
    
    stime = time.time()
    key, *exp_keys = jax.random.split(key, num_envs + 1)
    res = experiment_vmapped(jnp.stack(exp_keys), env_batch, algo_batch, N)
    elapsed_time = time.time() - stime
    print(f"Run time: {elapsed_time}s")
    
    cumulative_regret = np.cumsum(np.array(res["regret"].mean(0)))

    fig, axs = plt.subplots(1, 2)

    axs[0].set(xlabel="Samples", ylabel="Cumulative Regret")
    axs[0].plot(np.array(cumulative_regret))
    axs[0].set(xlabel="Samples", ylabel="Reward")
    axs[1].plot(np.array(res["reward"].mean(0)))
    plt.show()

