import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Chainbla-v0',
    entry_point='baselines.deepq.experiments.gym_chain.envs:ChainEnv',
)