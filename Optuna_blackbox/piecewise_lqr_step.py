import numpy as np

class piecewise_agent():
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
        cum_sumk = np.mean(sum_int[:,-40:],axis=-1)
        
        return cum_sumk

    def switch(self,init_ch, cum_sumk, LQR, S1, S2, S3):
        channel_arg = np.argsort(cum_sumk)[:,0:self.num_subband]
        
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
    
    def _2switch(self,init_ch, cum_sumk, LQR, S2, S3):
        channel_arg = np.argsort(cum_sumk)[:,0:self.num_subband]
        
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


    # def control_aware_subband_update(self,cum_sumk,K,norm_lqr):
    #     prob = np.zeros(K, dtype=np.float16)
    #     argmin = np.argmin(cum_sumk)
    #     prob[argmin] = norm_lqr
    #     prob[np.arange(len(prob))!=argmin] = (1 -prob[argmin]) * np.random.dirichlet(1/(cum_sumk[np.arange(len(prob))!=argmin]+1),size=1)[0]
    #     Ch_alloc = np.random.choice(np.arange(K),size = 1, p =prob)[0]
    #     #print(cum_sumk)
    #     return Ch_alloc

    def act_2switch(self, state, params, sumint, init_ch):
        LQR_int = state[:,0]
        #LQR_mean = state[:,1]

        S2 = params[0]
        S3 = params[1]
        sumint= self.int_measure(sumint)
        Ch_alloc = self._2switch(init_ch, sumint, LQR_int, S2, S3)
        return Ch_alloc
    
    def joint_act(self, state, params, sumint, init_ch):
        LQR_int = state[:,0]
        #LQR_mean = state[:,1]
        K_ = params[0]
        X0 = params[1]
        S2 = params[2]
        S3 = params[3]
        sumint= self.int_measure(sumint)
        P = 1/(1+np.exp(-K_*(LQR_int-X0)))
        Ch_alloc = self._2switch(init_ch, sumint, LQR_int, S2, S3)
        return P, Ch_alloc
    
    def power_act(self, state, params):
        LQR_int = state[:,0]
        #LQR_mean = state[:,1]
        K_ = params[0]
        X0 = params[1]
        P = 1/(1+np.exp(-K_*(LQR_int-X0)))
        return P
    
    def joint_act_w_penalty(self, state, params, sumint, init_ch):
        LQR_int = state[:,0]
        #LQR_mean = state[:,1]
        K_ = params[0]
        X0 = params[1]
        S2 = params[2]
        S3 = params[3]
        penalty = params[4]
        sumint= self.int_measure(sumint)
        P = 1/(1+np.exp(-K_*(LQR_int-X0)))
        Ch_alloc = self._2switch(init_ch, sumint, LQR_int, S2, S3)
        num_selected_subbands = np.sum(Ch_alloc,-1)
        P_penalize = P - penalty*num_selected_subbands
        Pt = np.maximum(0, P_penalize)

        return Pt, Ch_alloc

    
    def act(self, state, params, sumint, init_ch):
        LQR_int = state[:,0]
        #LQR_mean = state[:,1]
        S1= params[0]
        S2 = params[1]
        S3 = params[2]
        sumint= self.int_measure(sumint)
        Ch_alloc = self.switch(init_ch, sumint, LQR_int, S1, S2, S3)
        return Ch_alloc