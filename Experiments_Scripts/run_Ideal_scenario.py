import numpy as np
import matplotlib.pyplot as plt
from System_Environments.control_env import *
import torch
from System_Environments.static_infactory_env_wCartPole import env_subnetwork
import numpy as np
from System_Environments.control_env import *


def collect_mean_lqr(num_of_episodes, num_of_subnetworks, num_subbands, num_steps, factory_length, save_file_name):
        T_sensor = [1]
        instant_lqr__ = []
        aoi = []
        plants_state_x_error_ = []
        plants_state_theta_error_ = []
        plants_state_x_dot_error_ = []
        plants_state_theta_dot_error_ = []
        force__ = []
        labels = []
        total_num_prbs = 15

        subn_env = env_subnetwork(numCell=num_of_subnetworks, numSubbands=num_subbands, steps=num_steps, factoryarea=[factory_length, factory_length], numerology=5, RB_per_Subband = 2)
        subn_env.reset()
        subn_env.time_index = 0
        subn_env.pow_action = np.ones(num_of_subnetworks, dtype=np.float32)
        subn_env.chl_action = np.ones((num_of_subnetworks,num_subbands)) * np.array([[0,1,2]])
        mean_lqr__ = np.zeros((num_of_episodes, num_of_subnetworks))

        for num in range(num_of_episodes):
                instant_lqr_ = []
                env = WNCSEnv(ext_env=subn_env, n_UEs = num_of_subnetworks, Ideal_transmission = True)
                env.reset()
                
                num_selected_subband = [1 for i in range(num_of_subnetworks)]
                se__ = [1]
                plants_state_x_error = []
                plants_state_theta_error = []
                plants_state_x_dot_error = []
                plants_state_theta_dot_error = []
                force_ = []
                mean_lqr_ = []
                instant_lqr_ = []
                
                env.reset()
                lqr_ = []
                        
                for j in range(num_steps):
                        env._gen_sensor_data()
                        sinr__ = np.array([[[100,100,100] for i in range(num_subbands)] for i in range(num_of_subnetworks)])
                        subband_action = [[0,1,2] for i in range(num_of_subnetworks)]
                        reward, lqr, plants_state, force, buffer_size, sensor_control_aoi, control_sensor_aoi = env.run(sinr__,num_selected_subbands=num_selected_subband, subband_action=subband_action, num_RB_per_Subband= 2)
                        lqr_.append(lqr)
                        instant_lqr_.append(np.mean(lqr_))

                        aoi.append(sensor_control_aoi)
                mean_lqr__[num,:] = np.mean(lqr_, 0)
                print("Episode ", num)
        #x,y = generate_cdf(mean_lqr__, 1000)
        #plt.loglog(x,y)
        np.savez(save_file_name, mean_lqr = mean_lqr__)
        return mean_lqr__

if __name__ == '__main__':
        factory_length = 30
        num_steps = 1000
        num_of_subnetworks = 50
        num_of_episodes = 2
        num_subbands = 3
        save_file_name = 'IdealTransmission_mixedplant.npz'
        mean_lqr = collect_mean_lqr(num_of_episodes, num_of_subnetworks, num_subbands, num_steps, factory_length, save_file_name)
