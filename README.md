# Jax Bandits

Jax based library for multi-armed bandit problems.

Includes the following algorithms
- UCB1, UCB2
- Thompson Sampling
- Epsilon Greedy

## Installation

The package only depends on `jax` and `flax`. Follow instructions on those repositories for how to install

Then to install this package, simply


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
key, env_key = jax.random.split(key)

# First intialize a bandit environment e.g. Bernoulli Bandits which comes with the environment state and functions
key, env_key = jax.random.split(key)
env = BernoulliBandits.create(env_key, arms=16) # the env_key is the seed for randomly generating the arm probabilities

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
    # perform one update step in the algorithm. Provide RNG, algorithm state, and the environment. Note that since environments can change, an updated environment is returned as well
    algo_state, env, action, reward = algo.update_step(step_key, algo_state, env)
    # store the regret values
    regret = env.regret(action)
    regrets += [regret]
cumulative_regret = np.cumsum(np.array(regrets))
```

For a packaged, jitted version of the above loop, you can use the `experiment` function in the package

```python
from jaxbandits import experiment
res = experiment(key, env_state, algo_state, env.step, algo.update_step, steps=N)
cumulative_regret = np.cumsum(np.array(res["regret"]))
```

Due to the high volume of small operations, usually using the CPU backend will be faster. The GPU backend will be better if you plan to `vmap/pmap` the code, which is all possible as all of the algorithms and environments are registered as pytree nodes (via the `@flax.struct.dataclass` decorator). 

To run a batch of experiments, simply `vmap` the `experiment` function. Example code is provided in [examples/parallel.py]()

## Algos

The following algos are provided as so

```python
from jaxbandits import algos
algos.ThompsonSampling
algos.UCB1
algos.UCB2
algos.EpsilonGreedy

```