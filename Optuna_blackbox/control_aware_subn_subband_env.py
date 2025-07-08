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

    def __init__(self, n_subnetworks=25, n_subband=3, reward_type='sum', num_step = 100):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.n_subnetworks = n_subnetworks
        self.reward_type = reward_type
        self.n_subbands = n_subband
        self.num_step = num_step
        self.num_actions = self.n_subnetworks
        self.num_observation = self.n_subnetworks
        self.state_variables = 5
          
        self.source_env = env_subnetwork(numCell= self.n_subnetworks, group=self.n_subbands, problem='power', steps=self.num_step+1)
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
    
    def SISA(self,channel_info, num_of_subnetworks, num_of_channels):
        max_power = 1
        mask = np.eye(num_of_subnetworks)
        channel_info_intraIBS = [np.diag(channel_info).reshape((-1,1)) for i in range(num_of_channels)]
        channel_info_interIBS = [channel_info * (1-np.eye(num_of_subnetworks)) for i in range(num_of_channels)]
        w = np.zeros((num_of_subnetworks,num_of_subnetworks, num_of_channels))
        a_t = np.random.randint(num_of_channels, size=(num_of_subnetworks, 1))
        b_t = []
        for k in range(num_of_channels):
            w[:, :, k] = (channel_info_interIBS[k])/(np.tile(channel_info_intraIBS[k], (1, num_of_subnetworks)))
            b_t.append(np.where(a_t == k)[0])

        w_k = np.zeros((num_of_channels, 1))
        for _ in range(10):
            for n in range(num_of_subnetworks):
                for k in range(num_of_channels):
                    w_k[k] = np.sum(w[n, b_t[k], k]) + np.sum(w[b_t[k], n, k])

                a_t[n] = np.argmin(w_k)
                b_t = []
                for k in range(num_of_channels):
                    b_t.append(np.where(a_t == k)[0])

        subn_channel_index = a_t
    #print(a_t)
        return subn_channel_index
    
    def compute_obs(self):
        #des_ch = np.expand_dims(self.source_env.des_chgain,-1)
        lqr = np.expand_dims(self.source_env.lqr,-1)
        #print('lqr list', self.source_env.lqr_list)
        if self.time_index == 0:
            mean_lqr = np.expand_dims(self.source_env.lqr,-1)
        else:
            mean_lqr = np.expand_dims(np.mean(np.array(self.source_env.lqr_list),0),-1)
        #print('mean lqr',mean_lqr)
        bs = np.expand_dims(self.source_env.buffer_size,-1)
        aoi = np.expand_dims(self.source_env.sensor_control_aoi,-1)
        
        observation = np.concatenate((lqr, mean_lqr, bs, aoi), axis=-1, dtype=np.float32)
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

    def reset(self, seed=None, options='SISA'):
        super().reset(seed=seed)
        self.time_index = 0
        self.source_env.reset()
        #reset_action = self.action_space.sample()
        #print(reset_action)
        rxPow = self.source_env.rxPow[0:self.n_subnetworks,self.n_subnetworks:,0]
        if options == 'SISA':
            subband_action = np.squeeze(self.SISA(10**(np.negative(rxPow)/10), self.n_subnetworks, self.n_subbands))
        elif options == 'Random':
            subband_action = np.random.randint(self.n_subbands, size=self.n_subnetworks)
        else:
            raise NotImplementedError
        self.init_ch = np.tile(subband_action,(self.n_subbands,1)).T
        _ = self.source_env.simple_traffic_aware_single_channel_step(chl_action=np.tile(subband_action,(self.n_subbands,1)).T,time_index=0)
        #print('reset ',np.tile(self.init_ch,(self.n_subbands,1)))
        self.observation = self.compute_obs()
        cum_sumk = self.source_env.sum_int_history[:,:,0]
        #print(cum_sumk.shape)
        #print('init observation ',observation)
        #observation = self.check_return_obs(observation)
        self.info = {"rate":[], "action":[]}
        return self.observation, self.info, cum_sumk, subband_action

    def step(self, action):
        self.time_index = self.time_index + 1
        #valid_plants = np.arange(num_of_subnetworks)[env.activity_indicator[self.time_index].astype(bool)]
        if len(action.shape) == 1:
            action = np.tile(action,(self.n_subbands,1)).T
        self.source_env.subn_plant_control._gen_sensor_data()
        _ = self.source_env.simple_traffic_aware_single_channel_step(chl_action=action,time_index=self.time_index)
        #print('step ',action)
        self.observation = self.compute_obs()
        reward = self.reward_func(self.source_env.lqr)
        #print(self.time_index)
        truncated = False
        terminated = False
        if self.time_index == self.num_step:
            truncated = True
        self.info["rate"].append(self.source_env.sRate)
        self.info["action"].append(action)
        cum_sumk = self.source_env.sum_int_history #np.mean(self.source_env.sum_int_history[:,:,-10:], axis=-1)
        #print(self.source_env.sum_int_history[:,:,1:].shape)
        #print(cum_sumk.shape)
        # observation = observation.flatten()
        # observation = np.array(observation, dtype=np.float32)
        # observation = self.check_return_obs(observation)
        #print(observation)
        #print(observation.reshape(-1,2))
        return self.observation, reward, terminated, truncated, self.info, cum_sumk
