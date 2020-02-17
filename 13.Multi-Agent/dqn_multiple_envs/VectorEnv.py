import gym
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space

class VectorEnv(gym.Env):
    def __init__(self, num_envs, observation_space, action_space):
        super(VectorEnv, self).__init__()
        self.num_envs = num_envs
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = Tuple((action_space,) * num_envs)
        
        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_observation_space = observation_space
        self.single_action_space = action_space