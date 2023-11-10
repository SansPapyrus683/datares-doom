import gymnasium as gym
from stable_baselines3.ppo import PPO, policies
from stable_baselines3.common.env_util import make_vec_env

import vizdoom.gymnasium_wrapper  # just need the register commands
from obs_wrapper import ObservationWrapper


def wrap_env(env):
    """
    Create multiple environments: this speeds up training with PPO
    We apply two wrappers on the environment:
    1) The above wrapper that modifies the observations (takes only the image and resizes it)
    2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    This may lead to unstable learning, and we scale the rewards by 1/100
    """
    env = ObservationWrapper(env)
    env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    return env


# Training parameters
TRAINING_TIMESTEPS = 10 ** 6
N_STEPS = 128
N_ENVS = 8

envs = make_vec_env("VizdoomBasic-v0", n_envs=N_ENVS, wrapper_class=wrap_env)

agent = PPO(
    policies.CnnPolicy,
    envs,
    n_steps=N_STEPS,
    verbose=1
)

# Do the actual learning
# This will print out the results in the console.
# If agent gets better, "ep_rew_mean" should increase steadily
agent.learn(total_timesteps=TRAINING_TIMESTEPS)
