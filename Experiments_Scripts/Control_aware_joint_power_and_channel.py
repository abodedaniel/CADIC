import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from scipy.optimize import minimize, LinearConstraint
from scipy.io import savemat
from System_Environments.static_infactory_env_wCartPole import env_subnetwork
import time
np.set_printoptions(precision=0)

def objective_function(x,data,N,):
    p = x
    powerMat, _ = np.meshgrid(p,p, indexing = 'xy')
    mask = np.eye(N)
    data = powerMat*data
    receivePower = np.sum(data*mask,1)
    interP = data*(1-mask)
    intPower = np.sum(interP,1)
    sinr = np.divide(receivePower, intPower+ 1) 
    rate = np.log2(1+sinr)
    return np.negative(np.prod(rate))

def SISA(channel_info, num_of_subnetworks, num_of_channels):
    max_power = 1
    mask = np.expand_dims(np.eye(num_of_subnetworks),-1)
    channel_info_intraIBS = np.sum(channel_info*mask, 1)
    channel_info_interIBS = channel_info * (1-mask)
    w = np.zeros((num_of_subnetworks,num_of_subnetworks, num_of_channels))
    a_t = np.random.randint(num_of_channels, size=(num_of_subnetworks, 1))
    b_t = []
    for k in range(num_of_channels):
        w[:, :, k] = (channel_info_interIBS[:,:,k])/np.expand_dims(channel_info_intraIBS[:,k], -1)
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

def control_aware_sequential_greedy(Ch_mat,N,K,norm_lqr):
    Ch_alloc = []
    cum_sumk = np.zeros(K)
    for i in range(N):
        prob = np.zeros(K)
        # if i < K:
        #     Ch_alloc.append(i)
        #     continue
        for j in range(K):
            cum_sumk[j] = np.sum(Ch_mat[0:i,i][np.array(Ch_alloc) == j])
        argmin = np.argmin(cum_sumk)
        prob[argmin] = norm_lqr[i]
        prob[np.arange(len(prob))!=argmin] = (1 -prob[argmin]) * np.random.dirichlet(1/(cum_sumk[np.arange(len(prob))!=argmin]+1),size=1)[0]
        Ch_alloc.append(np.random.choice(np.arange(K),size = 1, p =prob)[0])
        #print(cum_sumk)
    return(np.array(Ch_alloc).flatten())

def sequential_greedy(Ch_mat,N,K):
    Ch_alloc = []
    cum_sumk = np.zeros(K)
    for i in range(N):
        if i < K:
            Ch_alloc.append(i)
            continue
        for j in range(K):
            cum_sumk[j] = np.sum(Ch_mat[0:i,i][np.array(Ch_alloc) == j])
        Ch_alloc.append(np.argmin(cum_sumk))
        #print(cum_sumk)
    return(Ch_alloc)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def control_aware_subband_update(cum_sumk,num_subn,K,norm_lqr):
    prob = np.zeros((num_subn, K), dtype=np.float16)
    argmin = np.argmin(cum_sumk, axis=-1, keepdims=True)
    #print(cum_sumk.shape)
    arg_rest = np.argsort(cum_sumk)[:,1:K]
    
    #print(argmin)
    np.put_along_axis(prob, argmin, np.expand_dims(norm_lqr,-1), axis=1)
    np.put_along_axis(prob, arg_rest, (1 - np.expand_dims(norm_lqr,-1)) * np.ones_like(arg_rest) * 0.5, axis=1)

    Ch_alloc = random_choice_prob_index(prob, axis=1)
    return Ch_alloc

# def control_aware_subband_update(cum_sumk,K,norm_lqr):
#     prob = np.zeros(K, dtype=np.float16)
#     argmin = np.argmin(cum_sumk)
#     prob[argmin] = norm_lqr
#     prob[np.arange(len(prob))!=argmin] = (1 -prob[argmin]) * np.random.dirichlet(1/(cum_sumk[np.arange(len(prob))!=argmin]+1),size=1)[0]
#     Ch_alloc = np.random.choice(np.arange(K),size = 1, p =prob)[0]
#     #print(cum_sumk)
#     return Ch_alloc

def switch(init_ch, cum_sumk, LQR, S1, S2, S3,num_subband):
    channel_arg = np.argsort(cum_sumk)[:,0:num_subband]
        
    subn_channel =  init_ch #np.zeros((K,num_subband))
    best_channel = channel_arg[:,0]
    #print('best channel ',best_channel)
    num_ch = np.piecewise(LQR, [LQR>=S1, LQR>=S2, LQR>=S3],[1,2,3])

    # To not switch
    to_not_switch = num_ch == 0
    subn_channel[to_not_switch,:] = np.expand_dims(init_ch[to_not_switch,0],-1)

    # To switch to 1st best channel
    to_switch_to_best_channel = num_ch == 1
    subn_channel[to_switch_to_best_channel,:] = np.expand_dims(best_channel[to_switch_to_best_channel],-1)

    # To switch to 2 best channel
    to_switch_to_two_channels = num_ch == 2
    subn_channel[to_switch_to_two_channels,0] = channel_arg[to_switch_to_two_channels,0]
    subn_channel[to_switch_to_two_channels,1:] = np.expand_dims(channel_arg[to_switch_to_two_channels,1],-1)

    # To switch to 3 best channel
    to_switch_to_3_channels = num_ch == 3
    subn_channel[to_switch_to_3_channels,:] = channel_arg[to_switch_to_3_channels,:]
    
    
    return subn_channel

def switch1(init_ch, cum_sumk, LQR, S2, S3,num_subband):
    channel_arg = np.argsort(cum_sumk)[:,0:num_subband]
        
    subn_channel =  init_ch #np.zeros((K,num_subband))
    best_channel = channel_arg[:,0]
    #print('best channel ',best_channel)
    num_ch = np.piecewise(LQR, [LQR>=S2, LQR>=S3],[2,3])

    # To switch to 1st best channel
    to_switch_to_best_channel = num_ch < 2
    subn_channel[to_switch_to_best_channel,:] = np.expand_dims(best_channel[to_switch_to_best_channel],-1)

    # To switch to 2 best channel
    to_switch_to_two_channels = num_ch == 2
    subn_channel[to_switch_to_two_channels,0] = channel_arg[to_switch_to_two_channels,0]
    subn_channel[to_switch_to_two_channels,1:] = np.expand_dims(channel_arg[to_switch_to_two_channels,1],-1)

    # To switch to 3 best channel
    to_switch_to_3_channels = num_ch == 3
    subn_channel[to_switch_to_3_channels,:] = channel_arg[to_switch_to_3_channels,:]
    return subn_channel

def logistic(L,K,X0,X):
    return(L/(1+np.exp(-K*(X-X0))))

length = 30
S1 = 4
S2 = 96  #for train 10, -0.4, 20, 96, 146  #for train 15 0.49, 16, 100, 186 #for train 20 (min max) 0.66, 14, 130, 246
S3 = 146

L = 1
K = 0.4
X0 = 20

K_0 = 0.2  #original 0.85
K_1 = 36
num_of_steps = 1000
num_of_subnetworks = 15
N_test = 15
num_of_episodes = 1000
run_id = 2
numerology=5 
RB_per_Subband = 3
num_subbands = 3
zerod_pow = 180 #-dBm
zerod_pow_ = 10**(-zerod_pow/10)

method = 'SISA' # 'SeqG' 'SISA', 'CASG', 'useall', 'Random', 'SISAwLQR', 'RandomwLQR', 'IdealSISA', 'superRandom','CICA-PC'
only_init_CA = True

if method == 'SISAwLQR' or method == 'RandomwLQR':
    name = 'counterzerod'+str(zerod_pow)+'Mixedplant3RBJointpowerswitch1multi_channel_TestN'+str(N_test)+str(K)+str(X0)+str(S2)+str(S3)+str(num_of_subnetworks)+'3030Run'+str(run_id)+method+'.npz'
elif method == 'CICA-PC':
    name = 'wTcounterMixedplant3RBpowerswitch'+str(K_0)+str(K_1)+str(num_of_subnetworks)+'3030Run'+str(run_id)+method+'.npz'
else:
    name = '10steps_2new_wTcounterw10GHzprodMixedplant3RBswitch_channel_'+str(num_of_subnetworks)+'3030Run'+str(run_id)+method+'.npz'

switch_channel = np.array([1,0])
switch_channel_prob = np.zeros((num_of_subnetworks,2))
switch_to_best_ch_prob = np.zeros((num_of_subnetworks,num_subbands))

plant_monitor = np.zeros((num_of_episodes, num_of_steps, num_of_subnetworks),dtype=np.uint8)
plant_states = np.zeros((num_of_episodes, num_of_steps, num_of_subnetworks,4))
plant_force = np.zeros((num_of_episodes, num_of_steps, num_of_subnetworks))
activity_indicator = np.zeros((num_of_episodes, num_of_steps, num_of_subnetworks),dtype=bool)
channel_use_history = np.zeros((num_of_episodes,num_of_steps,num_of_subnetworks,num_subbands), dtype=int)
sum_interference_history = np.zeros((num_of_episodes,num_of_subnetworks,num_subbands,num_of_steps))
ul_dl_error = np.zeros((num_of_episodes,num_of_subnetworks), dtype=int)
worst_BLER_mat = np.zeros((num_of_episodes, num_of_steps, num_of_subnetworks)) 
av_BLER_mat = np.zeros((num_of_episodes, num_of_steps, num_of_subnetworks))
ul_BLER = []
dl_BLER = []
ul_sinr = []
dl_sinr = []

transmit_power_allocated = np.zeros((num_of_episodes,num_of_steps,num_of_subnetworks))



power = []
subband_action = -1 * np.ones(num_of_subnetworks)
transmission_counter = 0

start = time.time()
env = env_subnetwork(numCell=num_of_subnetworks, numSubbands=num_subbands, steps=num_of_steps, factoryarea=[length, length], numerology=5, RB_per_Subband = RB_per_Subband)
lqr_list = []
dl_rate_list = []
ul_rate_list = []
power_list = []
for episode in range(num_of_episodes):
    env.reset()
    
    for step in range(num_of_steps):
        
        print(method + ' episode ', episode,' N=', num_of_subnetworks, ' run id=', run_id ,' step', step)
        env.subn_plant_control._gen_sensor_data()
        
        channel_lookup = np.zeros((num_of_subnetworks,2))
        switch_channel_prob = np.zeros((num_of_subnetworks,2))

        valid_plants = []
        count_failed = 0
        for n in range(num_of_subnetworks):
            if env.subn_plant_control.plants[n].tx_buffer['buffer_data'] == []:
                env.activity_indicator[step,n] = 0
            if env.subn_plant_control.plants[n].check_stop_condition():
                #env.activity_indicator[step,n] = 0
                plant_monitor[episode,step,n] = 1

        

        num_active = int(np.sum(env.activity_indicator[step]))
        active_rxPow = np.zeros((num_active, num_active))
        valid_plants = np.arange(num_of_subnetworks)[env.activity_indicator[step].astype(bool)]
        env.subn_plant_control.valid_plants =  np.argwhere(env.activity_indicator[step]).flatten().astype(int)    

        rxPow = np.mean(env.ul_rxPow[0:num_of_subnetworks,num_of_subnetworks:,:,:,step],axis=3)
        #print('valid plants ', valid_plants)
        rate_constraint_ = np.zeros(num_active)
        i = 0
        j = 0
        # for n in valid_plants:
        #     for m in valid_plants:
        #         #print(i,j)
        #         active_rxPow[int(i),int(j)] = np.mean(rxPow[int(n),int(m)])
        #         j += 1
        #     #rate_constraint_[i] = np.sum(env.subn_plant_control.plants[n].tx_buffer['buffer_data'])/(env.bw_to_PRBs[env.totalbw] * 180000) * 1e3
        #     i += 1
        #     j = 0
        
        #ch_gain_mat = (10**(np.negative(active_rxPow)/10)) #/ env.noisePower
        #SNR = np.diag(10**(np.negative(active_rxPow)/10)) / env.noisePower

        _,lqr_,_,_ = env.subn_plant_control._calc_lqr_cost()
        
        lqr_ = np.array(lqr_)
        #print('lqr ', lqr_)
        lqr = np.array(lqr_[env.activity_indicator[step].astype(bool)])

        N = len(env.subn_plant_control.valid_plants)

        #print('myvalid',env.subn_plant_control.valid_plants.flatten()) 
        #print('activity_ind',env.activity_indicator[step].flatten())   
        
        if step == 0:
            if method == 'SISA':
                subband_action = np.squeeze(SISA(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                pow = np.ones(num_of_subnetworks, dtype=np.float32)
            
            elif method == 'SeqG': 
                subband_action = np.squeeze(sequential_greedy(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                pow = np.ones(num_of_subnetworks, dtype=np.float32)
            
            elif method == 'SISA+PC':
                subband_action = np.squeeze(SISA(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                pow = np.ones(num_of_subnetworks)
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                unique_values, indices = np.unique(init_ch, return_inverse=True)
                unique_indices = {value: np.where(indices == i)[0].tolist() for i, value in enumerate(unique_values)}
                ch_gain = (10**(np.negative(rxPow)/10))/env.noisePower
                full_ch_gain = np.mean(ch_gain,-1)
                for k in unique_indices.values():
                    int_neighbour_ch_gain = full_ch_gain[np.ix_(k,k)]
                    pow[k] = scipy.optimize.fmin_slsqp(objective_function, x0= np.ones(len(k)), bounds= [(0,1)]*len(k),args=(int_neighbour_ch_gain,len(k)))
                    # print(k)
                    # print(pow[k])
                    # print(pow)
            elif method == 'IdealSISA':
                subband_action = np.squeeze(SISA(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                pow = np.ones(num_of_subnetworks, dtype=np.float32)
            elif method == 'SISAwLQR':
                subband_action = np.squeeze(SISA(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                pow = logistic(L,K,X0,lqr_) 
            elif method == 'Random':
                subband_action = np.random.randint(num_subbands, size=num_of_subnetworks)
                subband_action = np.tile(subband_action,(num_subbands,1)).T
                pow = np.ones(num_of_subnetworks, dtype=np.float32)
            elif method == 'superRandom':
                pow = np.random.rand(num_of_subnetworks)
                subband_action = np.random.randint(num_subbands, size=(num_of_subnetworks,num_subbands))

            elif method == 'RandomwLQR':
                subband_action = np.random.randint(num_subbands, size=num_of_subnetworks)
                subband_action = np.tile(subband_action,(num_subbands,1)).T
                pow = logistic(L,K,X0,lqr_) 
                # pow[pow < zerod_pow_] = 0
                # zerod_plants = list(np.where(pow == 0)[0])
                # for j in zerod_plants:
                #     env.subn_plant_control.plants[j].tx_buffer['buffer_data'] = []
                #     env.activity_indicator[step,j] = 0
            elif method == 'useall':
                subband_action = np.ones((num_of_subnetworks,num_subbands)) * np.array([[0,1,2]])
                pow = np.ones(num_of_subnetworks, dtype=np.float32)
            elif method == 'CICA-PC':
                subband_action = np.ones((num_of_subnetworks,num_subbands)) * np.array([[0,1,2]])
                pow = logistic(L,K_0,K_1,lqr_) 
            else:
                raise NotImplementedError
            
        elif method == 'CICA-PC' and step>0:
                subband_action = np.ones((num_of_subnetworks,num_subbands)) * np.array([[0,1,2]])
                pow = logistic(L,K_0,K_1,lqr_) 
        
        elif method == 'SISA' and step>0 and step%10 == 0:
                subband_action = np.squeeze(SISA(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                pow = np.ones(num_of_subnetworks, dtype=np.float32)

        elif method == 'SeqG' and step>0 and step%10 == 0:
                subband_action = np.squeeze(sequential_greedy(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                pow = np.ones(num_of_subnetworks, dtype=np.float32)
        
        elif method == 'SISA+PC' and step>0 and step%10 == 0:
                subband_action = np.squeeze(SISA(10**(np.negative(rxPow)/10), num_of_subnetworks, num_subbands))
                init_ch = subband_action
                current_channel = subband_action
                pow = np.ones(num_of_subnetworks)
                subband_action = np.tile(init_ch,(num_subbands,1)).T
                unique_values, indices = np.unique(init_ch, return_inverse=True)
                unique_indices = {value: np.where(indices == i)[0].tolist() for i, value in enumerate(unique_values)}
                ch_gain = (10**(np.negative(rxPow)/10))/env.noisePower
                full_ch_gain = np.mean(ch_gain,-1)
                for k in unique_indices.values():
                    int_neighbour_ch_gain = full_ch_gain[np.ix_(k,k)]
                    pow[k] = scipy.optimize.fmin_slsqp(objective_function, x0= np.ones(len(k)), bounds= [(0,1)]*len(k),args=(int_neighbour_ch_gain,len(k)))
                    
                    #print(pow)
        elif method == 'SISAwLQR' and step>0:
            cum_sumk = np.mean(env.sum_int_history[:,:,1:], axis=-1)

            #print('cum_sumk ', cum_sumk)
            LQR = lqr_ #np.mean(env.lqr_list,0)
            #subband_action = switch(current_channel, cum_sumk, LQR, S1, S2, S3, num_subbands)
            subband_action = switch1(current_channel, cum_sumk, LQR, S2, S3, num_subbands)
            pow = logistic(L,K,X0,lqr_)

        elif method == 'Random' and step>0:
            subband_action = np.random.randint(num_subbands, size=num_of_subnetworks)
            subband_action = np.tile(subband_action,(num_subbands,1)).T
            pow = np.ones(num_of_subnetworks, dtype=np.float32)
        
        elif method == 'superRandom' and step>0:
            pow = np.random.rand(num_of_subnetworks)
            subband_action = np.random.randint(num_subbands, size=(num_of_subnetworks,num_subbands))
        
        elif method == 'RandomwLQR' and step>0:
            cum_sumk = np.mean(env.sum_int_history[:,:,1:], axis=-1)

            #print('cum_sumk ', cum_sumk)
            LQR = lqr_ #np.mean(env.lqr_list,0)
            #subband_action = switch(current_channel, cum_sumk, LQR, S1, S2, S3, num_subbands)
            subband_action = switch1(current_channel, cum_sumk, LQR, S2, S3, num_subbands)
            pow = logistic(L,K,X0,lqr_)
            # pow[pow < zerod_pow_] = 0
            # zerod_plants = list(np.where(pow == 0)[0])
            # for j in zerod_plants:
            #     env.subn_plant_control.plants[j].tx_buffer['buffer_data'] = []
            #     env.activity_indicator[step,j] = 0

            
        else:
            pow = np.ones(num_of_subnetworks, dtype=np.float32)
            


        if np.any(lqr_ > 82):
            print(np.argwhere(lqr_ > 82))
         
            #print('subband_action',subband_action)  
        env.traffic_aware_joint_power_channel_step(pow, subband_action, step)
        current_channel = subband_action

        
        # else:
        #     if method == 'useall':
        #         subband_action = 6 * np.ones(num_active)
        #     elif method == 'SISA':
        #         subband_action = np.squeeze(SISA(ch_gain_mat, num_active, num_subbands))
        #     elif method == 'Random':
        #         subband_action = np.random.randint(num_subbands, size=num_active)
        #     elif method == 'CASG':
        #         subband_action = control_aware_sequential_greedy(ch_gain_mat,num_active,num_subbands,norm_lqr)
        #         print(subband_action)
        #     else:
        #         raise NotImplementedError
        #         print(subband_action)
        #     env.traffic_aware_multi_channel_step(subband_action, time_index=step)   
        ul_rate_list.append(env.ul_avRate) 
        dl_rate_list.append(env.dl_avRate)  
        channel_use_history[episode, step] = subband_action
        transmit_power_allocated[episode, step] = pow
        worst_BLER_mat[episode, step] = env.subn_plant_control.worst_ul_bler_mat
        av_BLER_mat[episode, step] = env.subn_plant_control.av_ul_bler_mat
        power_list.extend(env._tx_powers)


        #print('lqr',env.lqr) 
    ul_dl_error[episode,:] = env.subn_plant_control.ul_dl_error
    dl_BLER.extend(env.subn_plant_control.dl_BLER)
    ul_BLER.extend(env.subn_plant_control.ul_BLER)
    ul_sinr.extend(env.subn_plant_control.ul_sinr)
    dl_sinr.extend(env.subn_plant_control.dl_sinr)
    lqr_list.append(env.lqr_list)
    activity_indicator[episode,:,:] = env.activity_indicator
    sum_interference_history[episode] = env.sum_int_history[:,:,1:]
    transmission_counter = transmission_counter + env.subn_plant_control.transmission_counter
    
    print(transmission_counter)
    plant_states[episode] = np.array(env.plant_states_list) 
    plant_force[episode] = np.array(env.force_list)
    
        #print('lqr',env.lqr)
#print(np.mean(lqr_list))

    if episode%100 == 0 and episode > 100:
        np.savez(name, lqr = lqr_list, ul_dl_error=ul_dl_error, ul_BLER = ul_BLER, dl_BLER = dl_BLER, ul_sinr=ul_sinr, dl_sinr=dl_sinr,transmission_counter=[transmission_counter], power_list = power_list)


np.savez(name, lqr = lqr_list, ul_dl_error=ul_dl_error, ul_BLER = ul_BLER, dl_BLER = dl_BLER,ul_sinr=ul_sinr,  dl_sinr=dl_sinr, transmission_counter=[transmission_counter], power_list = power_list) #, tx_power_alloc = transmit_power_allocated, channel_use = channel_use_history, activity_indicator=activity_indicator, worst_BLER_mat=worst_BLER_mat, av_BLER_mat=av_BLER_mat)

# #np.savez(name, lqr = lqr_list, ulrate=ul_rate_list, dlrate=dl_rate_list, plant_states=plant_states, 
#          plant_monitor=plant_monitor, plant_force=plant_force, activity_indicator=activity_indicator, 
#          sum_int = sum_interference_history, channel_use = channel_use_history, ul_dl_error=ul_dl_error, 
#          dl_BLER = dl_BLER, ul_BLER = ul_BLER, dl_sinr=dl_sinr, ul_sinr=ul_sinr)

end = time.time()
print('runtime ', end - start)

