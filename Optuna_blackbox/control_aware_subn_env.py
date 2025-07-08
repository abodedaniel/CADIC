import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('../')
from static_infactory_env_wCartPole import env_subnetwork
#import torch
import warnings
import math
class SubnetworkEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, n_subnetworks=25, reward_type='sum', num_step = 100):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.n_subnetworks = n_subnetworks
        self.reward_type = reward_type
        self.n_subbands = 1
        self.num_step = num_step
        self.num_actions = self.n_subnetworks
        self.num_observation = self.n_subnetworks
        self.state_variables = 5
          
        self.source_env = env_subnetwork(numCell= self.n_subnetworks, group=1, problem='power', steps=self.num_step+1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_actions,))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.num_observation,self.state_variables), dtype=np.float32)
    
    def norm_action(self,action):
        lin_power = (action - self.action_space.low[0])/(self.action_space.high[0] - self.action_space.low[0])
        return lin_power
    
    def check_return_obs(self,obs):
        sum_violate = np.sum(obs>1) + np.sum(obs<0)
        if sum_violate > self.n_subnetworks/5:
            warnings.warn('Too many observation ' +str(sum_violate) +' is out of norm range')
        obs[obs>1] = 1
        obs[obs<0] = 0
        return obs
    
    def compute_obs(self):
        des_ch = np.expand_dims(self.source_env.des_chgain,-1)
        lqr = np.expand_dims(self.source_env.lqr,-1)
        #print('lqr list', self.source_env.lqr_list)
        if self.time_index == 0:
            mean_lqr = np.expand_dims(self.source_env.lqr,-1)
        else:
            mean_lqr = np.expand_dims(np.mean(np.array(self.source_env.lqr_list),0),-1)
        #print('mean lqr',mean_lqr)
        bs = np.expand_dims(self.source_env.buffer_size,-1)
        aoi = np.expand_dims(self.source_env.sensor_control_aoi,-1)
        
        observation = np.concatenate((des_ch, lqr, mean_lqr, bs, aoi), axis=-1, dtype=np.float32)
        return observation

    def reward_func(self, ind_reward):
        if self.reward_type == 'sum':
            ind_reward[ind_reward == math.inf] = 0 #np.max(ind_reward[ind_reward != math.inf])
            ind_reward[ind_reward == math.nan] = 0 #np.max(ind_reward[ind_reward != math.nan])
            reward = ind_reward #np.mean(np.array(ind_reward)>5)
        elif self.reward_type == 'worst':
            reward = np.max(ind_reward)
        elif self.reward_type == 'prod':
            reward = np.prod(ind_reward)
        else:
            raise NotImplementedError
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_index = 0
        self.source_env.reset()
        reset_action = self.action_space.sample()
        #print(reset_action)
        _ = self.source_env.simple_control_aware_centralized_power_step(action=reset_action, time_index=0)
        self.observation = self.compute_obs()
        #print('init observation ',observation)
        #observation = self.check_return_obs(observation)
        self.info = {"rate":[], "action":[]}
        return self.observation, self.info

    def step(self, action):
        self.time_index = self.time_index + 1
        self.source_env.subn_plant_control._gen_sensor_data()
        _ = self.source_env.simple_control_aware_centralized_power_step(action=action, time_index=self.time_index)
        self.observation = self.compute_obs()
        reward = self.reward_func(self.source_env.lqr)
        #print(self.time_index)
        truncated = False
        terminated = False
        if self.time_index == self.num_step:
            truncated = True
        self.info["rate"].append(self.source_env.sRate)
        self.info["action"].append(action)
        #print(observation)
        # observation = observation.flatten()
        # observation = np.array(observation, dtype=np.float32)
        # observation = self.check_return_obs(observation)
        #print(observation)
        #print(observation.reshape(-1,2))
        return self.observation, reward, terminated, truncated, self.info
