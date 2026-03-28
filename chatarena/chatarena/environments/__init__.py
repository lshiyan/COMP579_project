from ..config import EnvironmentConfig
from .base import ENV_REGISTRY, Environment, TimeStep, register_env
from .chameleon import Chameleon


# Load an environment from a config dictionary
def load_environment(config: EnvironmentConfig):
    try:
        env_cls = ENV_REGISTRY[config["env_type"]]
    except KeyError:
        raise ValueError(f"Unknown environment type: {config['env_type']}")

    env = env_cls.from_config(config)
    return env
