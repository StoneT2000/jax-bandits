import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxbandits import BernoulliBandits, algos, experiment

if __name__ == "__main__":

    # configure starting seed, number of parallel envs to run, number of arms, and max samples
    key = jax.random.PRNGKey(0)
    num_envs = 1024
    arms = 16
    N = 5000

    Algos = [algos.UCB1, algos.UCB2, algos.ThompsonSampling, algos.EpsilonGreedy]
    Envs = [BernoulliBandits]
    for Env in Envs:
        results = dict()
        for Algo in Algos:
            # vmap the env creation function and create num_envs envs with different states
            key, *env_keys = jax.random.split(key, num_envs + 1)
            env_batch = jax.vmap(Env.create, in_axes=[0, None])(
                jnp.stack(env_keys), arms
            )

            # vmap the algo creation function and create num_envs algos with different algo states
            algo_batch = jax.vmap(Algo.create, in_axes=[None], axis_size=num_envs)(arms)

            # vmap the experiment function
            experiment_vmapped = jax.vmap(
                experiment, in_axes=[0, 0, 0, None], axis_size=num_envs
            )

            stime = time.time()
            key, *exp_keys = jax.random.split(key, num_envs + 1)
            res = experiment_vmapped(jnp.stack(exp_keys), env_batch, algo_batch, N)
            compile_time = time.time() - stime
            print(f"Compile + Run time: {compile_time:.6f}s")
            results[Algo.__name__] = res

        fig = plt.figure(figsize=(14, 4))
        fig.subplots_adjust(bottom=0.15, wspace=0.1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig.suptitle(f"{Algo.__name__} results on {Env.__name__}")
        ax1.set(xlabel="Samples", ylabel="Avg. Cumulative Regret")
        
        ax2.set(xlabel="Samples", ylabel="Avg. Reward")
        
        for k in results:
            res = results[k]
            cumulative_regret = np.cumsum(np.array(res["regret"].mean(0)))
            ax1.plot(np.array(cumulative_regret), label=k)
            ax2.plot(np.array(res["reward"].mean(0)), label=k)
        
        plt.legend()
        plt.savefig(f"{Env.__name__}_results.png")