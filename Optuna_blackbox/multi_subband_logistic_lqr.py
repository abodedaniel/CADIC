import numpy as np

class logistic_agent():
    def __init__(self, n_subn, num_subband, Pmax):
        self.t = 0
        self.K = n_subn
        self.num_subband = num_subband
        self.Pmax = Pmax
        self.init_ch = 0

    def int_measure(self,sum_int):
        if len(sum_int.shape) == 2:
            sum_int = np.expand_dims(sum_int,-1)
            #cum_sumk = np.mean(sum_int,axis=-1)
        cum_sumk = np.mean(sum_int,axis=-1)
        
        return cum_sumk

    def switch(self,init_ch, cum_sumk, LQR, LQR_S0, LQR_S1):
        channel_arg = np.argsort(cum_sumk)[:,0:self.num_subband]
        
        subn_channel =  init_ch #np.zeros((K,num_subband))
        best_channel = channel_arg[:,0]
        #print('best channel ',best_channel)
        worst_channel = channel_arg[:,-1]
        #print('worst channel ', worst_channel)
        to_switch_for_improvement = LQR>LQR_S0 

        ## to switch to best channel
        to_switch_to_best_channel = np.logical_and(to_switch_for_improvement, LQR< LQR_S1)
        subn_channel[to_switch_to_best_channel,:] = np.expand_dims(best_channel[to_switch_to_best_channel],-1)

        ##to switch to 2 channels
        to_switch_to_two_channels = np.logical_and(to_switch_for_improvement, LQR >= LQR_S1)
        subn_channel[to_switch_to_two_channels,0] = init_ch[to_switch_to_two_channels,0] #the first channel is init
        subn_channel[to_switch_to_two_channels,1:] = np.expand_dims(best_channel[to_switch_to_two_channels],-1) #the second channel is best
        #if best channel is same as init channel, switch to 2nd best channel
        subn_channel[np.where(subn_channel[:,1] == subn_channel[:,0], to_switch_to_two_channels, False),1:] = np.expand_dims(channel_arg[np.where(subn_channel[:,1] == subn_channel[:,0], to_switch_to_two_channels, False),1],-1)
        
        ## to switch to worst channel
        # to_switch_to_worst_channel = np.logical_and(np.invert(to_switch_for_improvement), LQR< LQR_S2)
        # subn_channel[to_switch_to_worst_channel,:] = np.expand_dims(worst_channel[to_switch_to_worst_channel],-1)

        # ## to go back to init channel
        to_switch_to_init_channel = np.invert(to_switch_for_improvement) #np.logical_and(np.invert(to_switch_for_improvement), LQR >= LQR_S2)
        subn_channel[to_switch_to_init_channel,:] = np.expand_dims(init_ch[to_switch_to_init_channel,0],-1)
        return subn_channel


    # def control_aware_subband_update(self,cum_sumk,K,norm_lqr):
    #     prob = np.zeros(K, dtype=np.float16)
    #     argmin = np.argmin(cum_sumk)
    #     prob[argmin] = norm_lqr
    #     prob[np.arange(len(prob))!=argmin] = (1 -prob[argmin]) * np.random.dirichlet(1/(cum_sumk[np.arange(len(prob))!=argmin]+1),size=1)[0]
    #     Ch_alloc = np.random.choice(np.arange(K),size = 1, p =prob)[0]
    #     #print(cum_sumk)
    #     return Ch_alloc
    
    def act(self, state, params, sumint, init_ch):
        
        LQR_int = state[:,0]
        #LQR_mean = state[:,1]
        LQR_S0= params[0]
        LQR_S1 = params[1]
        sumint= self.int_measure(sumint)
        Ch_alloc = self.switch(init_ch, sumint, LQR_int, LQR_S0, LQR_S1)
        return Ch_alloc