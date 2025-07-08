#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:48:12 2021

@author: root
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import bisect
from scipy import special as sp

##############################################################################################################################################################
class RRScheduler:
    def __init__(self, K):
        self.K = K
        self.t = 0
    def select_action(self,):
        action = self.t % self.K
        self.t += 1
        return action

class RRIfTrafficScheduler:
    def __init__(self, K):
        self.K = K
        self.t = 0

    def check_traffic(self, plant):
            return True if plant.tx_buffer['buffer_data'] > 0 else False

    def select_action(self, plants):
        for i in range(self.K):
            action = self.t % self.K
            if self.check_traffic(plants[action]):
                self.t += 1
                return action
            else:
                self.t += 1
        return 0

class AgeScheduler:
    def __init__(self,):
        self.test = 0
    def select_action(self, ul_aoi):
        return np.argmax(ul_aoi)

class CqiScheduler:
    def __init__(self, K):
        self.K = K

    def check_traffic(self, plant):
        return True if plant.tx_buffer['buffer_data'] > 0 else False

    def select_action(self, plants, option):
        plant_cqi = []
        for i in range(self.K):
            plant_cqi.append((i, plants[i].cqi))
        reverse = False if option=='min' else True
        plant_cqi.sort(key=lambda x:x[1], reverse=reverse)
        for (i, j) in plant_cqi:
            if self.check_traffic(plants[i]):
                return i
        return 0

##############################################################################################################################################################
class Controller():
    def __init__(self, optim_k,
                 T_gen = 1, data_size = 16*8):
        self.optim_k = np.expand_dims(np.array(optim_k),0) # Calculated using lqr() function of MATLAB
        self.num_states = 4
        self.data_size = data_size

        # TX/RX buffer
        self.tx_buffer = {'control': 0.0, 'T':0}
        self.rx_buffer = {'state': np.zeros(self.num_states), 'T': 0,}

        # TTI of first generation and transmission
        self.tti_next_gen = 0
        self.tti_next_comm = 0

    def update_tx_buffer(self, ref_state, t):
        # LQR Control
        cur_state = self.read_states()
        error = ref_state.reshape((4,1)) - cur_state.reshape((4,1))
        #print(error)
        # if np.any(error > 10) or np.any(error < -10):
        #     self.tx_buffer['control'] = 1
        # else:
        self.tx_buffer['control'] = np.dot(self.optim_k, error)[0,0]
        # if np.isnan(self.tx_buffer['control']):
        #     #print(cur_state)
        self.tx_buffer['T'] = t

    def update_rx_buffer(self, rx_data):
        self.rx_buffer.update(rx_data)

    def read_states(self,aoi=None, actual_state=None):
        return list(self.rx_buffer.values())[0]

##############################################################################################################################################################
class CartPole():
    def __init__(self, tau=0.001, ref_state=np.array([0.0, 0.0, 0.0, 0.0]),
                 T_gen = 1, D=64*8, cqi=5, length=0.05):
        self.gravity = 9.8
        self.masscart = 0.5
        self.masspole = 0.1
        self.length = length  # actually half the pole's length
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.num_states = 4
        self.cqi = cqi
        self.data_size = D # data size (in bits) for transmission
        self.tau = tau  # seconds between state updates
        self.T_gen = T_gen

        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        self.x_threshold = 3.0                                       
        self.ref_state = ref_state # desired state
        self.state = np.random.uniform(low=-0.2, high=0.2, size=(4,)) #Is this a good initialization for x, and theta

        # TX/RX buffer
        self.tx_buffer = {'state': [], 'T': [], 'buffer_data': []}
        self.rx_buffer = {'control': 0.0,}

        # TTI of first generation and communication
        self.tti_next_gen = 0
        self.tti_next_comm = 0

    def state_update(self,):                                #How does the state change when no control force is applied, should rx_buffer content change to 0   
        if np.abs(self.state[2]) > 2*math.pi:
            self.state[2] = 0
        if np.abs(self.state[0]) > 100:
            self.state[0] = 100 * np.sign(self.state[0]) 
        if np.abs(self.rx_buffer['control']) > 1000:
            self.rx_buffer['control'] = 1000 * np.sign(self.rx_buffer['control'])

        x, x_dot, theta, theta_dot = self.state
        force = self.rx_buffer['control']
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot #+ np.random.uniform(-0.01, 0.01, 1)
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

    def check_stop_condition(self,):
        done = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2]  < -self.theta_threshold_radians
            or self.state[2]  > self.theta_threshold_radians
        )
        return done

    def update_tx_buffer(self, t):
        noise = 0.05
        
        if self.finish_tranmission == True:
            self.tx_buffer['buffer_data'].append(self.data_size) 
            self.tx_buffer['T'].append(t)
            self.tx_buffer['state'].append(self.state + np.random.uniform(-noise, noise, self.num_states))
        else:
            self.tx_buffer['buffer_data'] = self.data_size
            self.tx_buffer['T'] = t
            self.tx_buffer['state'] = self.state + np.random.uniform(-noise, noise, self.num_states)

    def update_rx_buffer(self, rx_data):
        self.rx_buffer.update(rx_data)

    def update_next_gen_time(self, l=None):
        if l is not None:
            self.tti_next_gen = self.tti_next_gen + l
        else:
            self.tti_next_gen = self.tti_next_gen + self.T_gen

    def update_next_comm_time(self, t, l):
        self.tti_next_comm = t + l

##############################################################################################################################################################
class WNCSEnv():

    def __init__(self,ext_env, n_UEs=1, T_sensor=[1,3], D=[64*8], length=[0.05,0.1], control_D = 32*8, finish_tranmission=False, transmit_error_model=True, UL_Tx_latency=1e-4, bw_per_RB= 12 * 480e3, Ideal_transmission=False):
        super().__init__()
        self.K = n_UEs
        self.T_sensor = T_sensor
        self.finish_tranmission = finish_tranmission
        self.transmit_error_model = transmit_error_model
        self.data_size = D
        self.buffer_data_history =  np.zeros(n_UEs)
        self.length = length
        self.length_to_optim_k = {0.5:[-3.1623,  -12.8616, -107.2509,  -42.0116], 0.1:[-3.1623, -12.6073, -97.2166, -33.5452], 
                                  0.05:[-3.1623, -12.5765, -96.0126, -32.5741],
                                  0.01:[-3.1623, -12.5520, -95.0587, -31.8114]}
        self.control_data_size = control_D
        self.Ideal_transmission = Ideal_transmission
        #can also select different polelengths or mass
        self.cqi = [1, 3, 5]
        self.UL_Tx_latency = UL_Tx_latency # (s)
        self.valid_plants = np.arange(self.K, dtype=np.int32)
        self.Q = np.array([[1,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,100]])
        self.R = 0.1
        self.ext_env = ext_env
        self.transmission_counter = 0
        
        #self.activity_indicator = np.ones(n_UEs)
        self.buffer_size = np.zeros(n_UEs)
        self.bw_per_RB = bw_per_RB
        #self.N_PRBs = self.bw_to_PRBs[BW] # which depends on the total BW
        self.min_rate = np.zeros(self.K)
        self.ul_BLER = [] 
        self.dl_BLER = []
        self.ul_sinr = [] 
        self.dl_sinr = []

    def reset(self,):

        self.t = 0 # timestamp
        self.t_count = np.zeros(self.K) # counter for when transmission occur
        self.num_data_gen =  np.zeros(self.K)
        self.num_data_trans = np.zeros(self.K)
        self.sensor_control_aoi = np.zeros(self.K)
        self.control_sensor_aoi = np.zeros(self.K)
        self.ul_dl_error = np.zeros(self.K)
        

        self.current_RB = 0
        self.sent_data = np.zeros(self.K)
        length_  = [np.random.choice(self.length) for i in range(self.K)]
        T_sensor_length = {0.05:1, 0.1:3}

        # Creating K different control systems
        self.plants = [CartPole(T_gen= T_sensor_length[length_[i]], D=random.choice(self.data_size), cqi=random.choice(self.cqi), length = length_[i])
                       for i in range(self.K)]
    
        #print('buffer data ',self.plants[0].tx_buffer['buffer_data'])
        self.controllers = [Controller(optim_k = self.length_to_optim_k[self.plants[i].length], data_size=self.control_data_size)
                            for i in range(self.K)]
        
        for i in range(self.K):
            self.plants[i].finish_tranmission = self.finish_tranmission
            self.plants[i].tti_next_comm = np.random.randint(self.plants[i].T_gen)
            self.plants[i].tti_next_gen = self.plants[i].tti_next_comm
            #self.min_rate[i] = (self.plants[i].data_size / self.plants[i].T_gen)/(180000 * 15) * 1e3
        

    def step(self,):
        pass

    def run(self, sinr_i_k, num_selected_subbands, subband_action, num_RB_per_Subband):
        active_dl = np.zeros(self.K)
        self.av_ul_bler_mat = np.zeros(self.K)
        self.worst_ul_bler_mat = np.zeros(self.K)
        for i in range(self.K): #i is plant id
            subbands = np.unique(subband_action[i]).astype(int)
            sinr_i_k_ = sinr_i_k[i,subbands,:].flatten()
            active_dl[i] = self._sensor_control_comm(i, sinr_i_k_, num_selected_subbands[i], num_RB_per_Subband)

            if self.finish_tranmission == True:
                self.buffer_size[i] = np.sum(self.plants[i].tx_buffer['buffer_data'])
            else:
                self.buffer_size[i] = self.plants[i].data_size
        
        dl_sinr_i_k = self.ext_env.dl_traffic_aware_joint_power_channel_sinr_calc(active_dl)
        for i in list(np.nonzero(active_dl)[0]): #i is plant id
            subbands = np.unique(subband_action[i]).astype(int)
            dl_sinr_i_k_ = dl_sinr_i_k[i,subbands,:].flatten()
            self._gen_control_data(i, dl_sinr_i_k_ , num_selected_subbands[i], num_RB_per_Subband)

        

        #self.current_RB += 1
        # for i in range(self.K):
        #     print(self.plants[i].tx_buffer['buffer_data'])
        #     print(self.plants[i].T_gen)
        #reward = 0.0
        #print(self.sensor_control_aoi)
        #if self.current_RB % self.N_PRBs == 0:
        self._update_metrics('run')
        self._plant_state_update()
        
        # print('t = ',self.t)
        reward, lqr, plants_state, force = self._calc_lqr_cost()


        return reward, lqr, plants_state, force, self.buffer_size, self.sensor_control_aoi, self.control_sensor_aoi

    def _gen_sensor_data(self,):
        for i in range(self.K):
            if self.t == self.plants[i].tti_next_gen:
                self.num_data_gen[i] = self.num_data_gen[i] + 1
                self.plants[i].update_tx_buffer(self.t)
                self.buffer_data_history[i] = self.buffer_data_history[i] + self.plants[i].data_size 
                self.plants[i].update_next_gen_time()
            #print('plant ', i,'data to transmit',self.plants[i].tx_buffer)

    def _gen_control_data(self, i, sinr_k_ , num_selected_subbands, num_RB_per_Subband):
        ref_state = self.plants[i].ref_state
        self.controllers[i].update_tx_buffer(ref_state, self.t)
        dl_block_error_p = self._block_error_prob(sinr_k_, self.controllers[i].data_size, self.UL_Tx_latency, num_selected_subbands, num_RB_per_Subband)
        self.dl_sinr.extend(list(sinr_k_))
        self.dl_BLER.extend(list(dl_block_error_p))
        TX_success = self._check_TX(dl_block_error_p)
        if self.Ideal_transmission:
                TX_success = True
        if TX_success:
            self._control_sensor_comm(i)
        else: 
            self.ul_dl_error[i] = self.ul_dl_error[i] + 1

    def proc_data(self,p,txbit):
        bd = self.plants[p].tx_buffer['buffer_data']
        zeroth_pos = -1
        for i in range(len(bd)):
            d = bd[i] - txbit
            #print(d)
            if d > 0:
                bd[i] = d
                break
            elif d == 0:
                bd[i] = d
                zeroth_pos= zeroth_pos+1
                break
            else:
                bd[i] = 0
                zeroth_pos= zeroth_pos+1
                txbit = abs(d)
        self.plants[p].tx_buffer['buffer_data'] = bd
        return zeroth_pos


    def _sensor_control_comm(self, i, sinr_k_, num_selected_subbands, num_RB_per_Subband):
        if self.finish_tranmission:
            if self.plants[i].tx_buffer['buffer_data'] != []:
                #self.activity_indicator[i] = 1
                #tx_bits = self._calc_tx_bits(se, total_num_prbs, num_subbands, num_selected_subbands)

                zeroth_pos = self.proc_data(i, tx_bits)

                #print('')
                #print('###############################')
                #print('plant ',i)
                #print('buffer size history',self.buffer_data_history[i])
                #print('tx bits', tx_bits)
                #print('sent data ',self.sent_data[i])
                
                if zeroth_pos>-1:
                    #print('I completed transmission')
                    self.t_count[i] += 1
                    received_data_state = self.plants[i].tx_buffer['state'].pop(zeroth_pos)
                    received_data_T = self.plants[i].tx_buffer['T'].pop(zeroth_pos) 
                    bz = self.plants[i].tx_buffer['buffer_data'].pop(zeroth_pos)

                    del self.plants[i].tx_buffer['state'][0:zeroth_pos]
                    del self.plants[i].tx_buffer['T'][0:zeroth_pos]
                    del self.plants[i].tx_buffer['buffer_data'][0:zeroth_pos]
                    received_data = {}
                    received_data['state'] = received_data_state
                    received_data['T'] = received_data_T 
                    
                    #print('received data',received_data)
                    self.controllers[i].update_rx_buffer(received_data)
                    self._gen_control_data(i)
                    self._update_metrics('sensor_control', received_data['T'], i)
                #print('t count', self.t_count[i])
                #print('num data gen', self.num_data_gen[i])
                #print('num data transmitted', self.num_data_trans[i])
        elif self.transmit_error_model:
            if self.plants[i].tx_buffer['buffer_data'] != []:
                self.transmission_counter = self.transmission_counter +1
                block_error_p = self._block_error_prob(sinr_k_, self.plants[i].tx_buffer['buffer_data'], self.UL_Tx_latency, num_selected_subbands, num_RB_per_Subband)
                self.av_ul_bler_mat[i] = np.mean(block_error_p)
                self.worst_ul_bler_mat[i] = np.max(block_error_p)
                self.ul_BLER.extend(list(block_error_p))
                self.ul_sinr.extend(list(sinr_k_))
                TX_success = self._check_TX(block_error_p)
                if self.Ideal_transmission:
                    TX_success = True
                #print(TX_success)
                if TX_success:
                    received_data = self.plants[i].tx_buffer
                    self.controllers[i].update_rx_buffer(received_data)
                    self._update_metrics('sensor_control', received_data['T'], i)
                    self.plants[i].tx_buffer['state'] = []
                    self.plants[i].tx_buffer['T'] = []
                    self.plants[i].tx_buffer['buffer_data'] = []
                    return 1
                else:
                    return 0
            else:
                return 0

                    
        else:
            if self.plants[i].tx_buffer['buffer_data'] > 0:
                tx_bits = self._calc_tx_bits(se, total_num_prbs, num_subbands, num_selected_subbands)
                self.plants[i].tx_buffer['buffer_data'] -= tx_bits
                if self.plants[i].tx_buffer['buffer_data'] <= 0:
                    received_data = self.plants[i].tx_buffer
                    self.controllers[i].update_rx_buffer(received_data)
                    self._gen_control_data(i)
                    self._update_metrics('sensor_control', received_data['T'], i)

    def _control_sensor_comm(self, i):
        received_data = self.controllers[i].tx_buffer
        self.plants[i].update_rx_buffer(received_data)
        self._update_metrics('control_sensor', received_data['T'], i)

    def _plant_state_update(self,):
        for i in range(self.K):
            self.plants[i].state_update()

    def _block_error_prob(self,sinr_per_RB, data_sz, latency, num_of_selected_subbands, num_RB_per_Subband):
        N = 2 * self.bw_per_RB * latency
        b = (2 * data_sz)/(num_RB_per_Subband*num_of_selected_subbands)
        C = 0.5 * np.log2(1 + sinr_per_RB)
        V = ((sinr_per_RB*(sinr_per_RB+2))/(2* ((sinr_per_RB +1)**2))) * (np.log2(math.e))**2
        x = (N*C - b + 0.5* np.log2(N))/np.sqrt(N*V)
        block_error_p = 0.5-0.5*sp.erf(x/math.sqrt(2))
        return block_error_p
    
    def _check_TX(self, block_error_p):
        p = random.uniform(0,1)
        TX_success = False
        if np.all(p > block_error_p):
            TX_success = True
        return TX_success

    def _calc_tx_bits(self, se, total_num_prbs, num_subbands, num_selected_subbands):
        
        #print('num selected subband ', num_selected_subbands, ' | allocated bandwidth ', (bw_per_RB * (total_num_prbs/num_subbands)*(num_selected_subbands)), ' | se ', se)
        return (self.bw_per_RB * (total_num_prbs/num_subbands)*(num_selected_subbands))  * se * self.UL_Tx_latency # Number of bits transmitted in Tx_time = 0.1 ms

    def _calc_lqr_cost(self):
        lqr = []
        plants_state = []
        force =  []
        for i in range(self.K):
            lqr.append(np.sum(self.plants[i].state.T * self.Q * self.plants[i].state) + np.sum(self.plants[i].rx_buffer['control']**2 * self.R)) #This LQR cost seems to miss the weight Q and R 
            plants_state.append(self.plants[i].state)
            force.append(self.plants[i].rx_buffer['control'])
        mean_lqr = np.mean(np.array(lqr))
        return mean_lqr, lqr, plants_state, force

    def _update_metrics(self, metric_type, T=None, index=None):
        if metric_type == 'run':
            self.t += 1
            self.sensor_control_aoi += 1
            self.control_sensor_aoi += 1
        elif metric_type == 'sensor_control':
            self.sensor_control_aoi[index] = self.t - T
        elif metric_type == 'control_sensor':
            self.control_sensor_aoi[index] = self.t - T
        else:
            raise ValueError('Metric Type not specified')

# if __name__ == "__main__":

#     env = WNCSEnv(n_UEs=10, BW=20)

#     schedulers = {'RR': RRScheduler(K=env.K), 'RRIfTraf': RRIfTrafficScheduler(K=env.K), 'AgeSched': AgeScheduler(),
#              'minCQI': CqiScheduler(K=env.K), 'maxCQI': CqiScheduler(K=env.K)}

#     results = {}

#     for agent in schedulers:
#         print(f'Starting agent: {agent}')
#         results[agent] = []
#         n_eps = 10
#         n_env_steps = 1000

#         for e in range(n_eps):
#             if e % 20 == 0:
#                 print(f'   Starting Epside: {e}')
#             env.reset()
#             mean_lqr = 0.0
#             while env.t < n_env_steps:
#                 if agent == 'RR':
#                     action = schedulers[agent].select_action()
#                 elif agent == 'RRIfTraf':
#                     action = schedulers[agent].select_action(env.plants)
#                 elif agent == 'AgeSched':
#                     action = schedulers[agent].select_action(env.sensor_control_aoi)
#                 elif agent == 'minCQI':
#                     action = schedulers[agent].select_action(env.plants, 'min')
#                 elif agent == 'maxCQI':
#                     action = schedulers[agent].select_action(env.plants, 'max')
#                 else:
#                     raise ValueError('Agent not specified.')

#                 reward = env.run(action)

#                 mean_lqr += reward

#             results[agent].append(mean_lqr)

