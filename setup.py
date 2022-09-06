import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="jaxbandits",
    version="1.0.0",
    author="Stone Tao",
    description="A Jax based library with multi-armed bandit algorithms and environments",
    license="MIT",
    keywords=["reinforcement-learning", "machine-learning", "ai", "bandits"],
    url="https://github.com/StoneT2000/jax-bandits",
    packages=["jaxbandits"],
    long_description=read("README.md"),
)
