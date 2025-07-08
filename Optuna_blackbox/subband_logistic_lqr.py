import numpy as np

class logistic_agent():
    def __init__(self, n_subn, num_subband, Pmax, subband_action):
        self.t = 0
        self.K = n_subn
        self.num_subband = num_subband
        self.Pmax = Pmax
        self.current_channel = subband_action

    def random_choice_prob_index(self,a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)

    def int_measure(self,sum_int):
        if len(sum_int.shape) == 2:
            sum_int = np.expand_dims(sum_int,-1)
        cum_sumk = np.mean(sum_int[:,:,-100:],axis=-1)
        return cum_sumk
    
    def control_aware_subband_update(self,cum_sumk,num_subn,K,norm_lqr):
        prob = np.zeros((num_subn, K), dtype=np.float16)
        argmin = np.argmin(cum_sumk, axis=-1, keepdims=True)
        arg_rest = np.argsort(cum_sumk)[:,1:K]
        #print(cum_sumk)
        #print(argmin)
        np.put_along_axis(prob, argmin, np.expand_dims(norm_lqr,-1), axis=1)
        np.put_along_axis(prob, arg_rest, (1 - np.expand_dims(norm_lqr,-1)) * np.ones_like(arg_rest) * 0.5, axis=1)

        Ch_alloc = self.random_choice_prob_index(prob, axis=1)
        return Ch_alloc

    # def control_aware_subband_update(self,cum_sumk,K,norm_lqr):
    #     prob = np.zeros(K, dtype=np.float16)
    #     argmin = np.argmin(cum_sumk)
    #     prob[argmin] = norm_lqr
    #     prob[np.arange(len(prob))!=argmin] = (1 -prob[argmin]) * np.random.dirichlet(1/(cum_sumk[np.arange(len(prob))!=argmin]+1),size=1)[0]
    #     Ch_alloc = np.random.choice(np.arange(K),size = 1, p =prob)[0]
    #     #print(cum_sumk)
    #     return Ch_alloc
    
    def act(self, state, params, sumint):
        channel_lookup = np.zeros((self.K,2))
        switch_channel_prob = np.zeros((self.K,2))
        LQR_int = state[:,0]
        K_ = 1 #params[0]
        X0 = params[0]
        P = 1/(1+np.exp(-K_*(LQR_int-X0)))
        cum_sumk = self.int_measure(sumint)
        channel_lookup[:,0] = np.argmin(cum_sumk, axis=-1)
        channel_lookup[:,1] = self.current_channel
        switch_channel_prob[:,0] = P>0.5
        switch_channel_prob[:,1] = 1 - (P>0.5)
        sel = self.random_choice_prob_index(switch_channel_prob, axis=1)
        Ch_alloc = channel_lookup[np.arange(self.K),sel].astype(int)
        self.current_channel = Ch_alloc
        return Ch_alloc