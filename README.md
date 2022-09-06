# Jax Bandits

Jax based library for multi-armed bandit problems.

Includes the following algorithms
- UCB1, UCB2
- Thompson Sampling
- Epsilon Greedy


## Usage

This library provides a simple jax based environment interface for Multi-armed bandits as well as algorithms and pure function paradigm of jax. Created objects (like environments or algorithms) don't hold any state, only configurations/hyper parameters and jitted pure functions.

The following shows how to initialize an environment and an algorithm.

```python
import jax
import numpy as np
from jaxbandits import BernoulliBandits, algos

# set backend to CPU as usually it's faster due to the dispatch overhead on the GPU. 
jax.config.update('jax_platform_name', 'cpu')

key = jax.random.PRNGKey(0)
key, reset_key = jax.random.split(key)

# First intialize a bandit environment e.g. Bernoulli Bandits and get the environment state
env = BernoulliBandits(arms=16)
# the reset_key is the seed for randomly generating the arm probabilities
env_state = env.reset(key=reset_key)

# Then we initialize an algorithm e.g. Thompson Sampling and get the algorithm state
algo = algos.ThompsonSampling(env.arms)
algo_state = algo.reset()

```

To then start experimenting and solving, run

```python
N = 4096
regrets = []
for i in range(N):
    key, step_key = jax.random.split(key)
    # perform one update step in the algorithm. Provide RNG, algorithm state, env state, and the env step function
    algo_state, env_state, action, reward = algo.update_step(step_key, algo_state, env_state, env.step)
    # store the regret values
    regret = np.max(env_state.arm_probs) - env_state.arm_probs[action]
    regrets += [regret]
cumulative_regret = np.cumsum(np.array(regrets))
```

For a packaged, jitted version of the above loop, you can use the `experiment` function in the package

```python
from jaxbandits import experiment
res = experiment(key, env_state, algo_state, env.step, algo.update_step, steps=N)
cumulative_regret = np.cumsum(np.array(res["regret"]))
```

Due to the high volume of small operations, usually using the CPU backend will be faster. The GPU backend will be better if you plan to `vmap/pmap` the code.

## Algos

The following algos are provided as so

```python
from jaxbandits import algos
algos.ThompsonSampling
algos.UCB1
algos.UCB2
algos.EpsilonGreedy

```