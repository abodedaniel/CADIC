import numpy as np

class logistic_agent():
    def __init__(self, n_subn, Pmax):
        self.t = 0
        self.K = n_subn
        self.Pmax = Pmax


    def act(self, state, params):
        LQR_int = state[:,1]
        K_ = params[0]
        X0 = params[1]
        P = 1/(1+np.exp(-K_*(LQR_int-X0)))
        return P