import optuna
import matplotlib.pyplot as plt
from subband_logistic_lqr import logistic_agent
from control_aware_subn_subband_env import SubnetworkEnv
import numpy as np
import time

global n_subnetworks, n_eps, n_steps, n_subband
n_eps = 10
n_steps = 3000
n_subnetworks = 100
n_subband = 3
n_trials = 200

start = time.time()
def objective(trial):
    #x1 = trial.suggest_float('x1', 0, 1, step=0.1)
    x2 = trial.suggest_int('x2', 2, 2000, step=2)
    
    env = SubnetworkEnv(num_step=n_steps, n_subnetworks=n_subnetworks, n_subband = n_subband)       
    rwd = np.zeros((n_eps, n_steps,env.n_subnetworks))      
    

    for ep in range(n_eps):  # Run episodes
        #env.seed(seed=np.random.randint(999999999))
        reward = 0
        done = False
        state, info, sum_int, subband = env.reset()
        agent = logistic_agent(n_subn=env.n_subnetworks, num_subband= n_subband, Pmax=0, subband_action=subband)
        #print(sum_int)
        for t in range(n_steps):  # Run one episode
            action = agent.act(state,[x2],sum_int) #np.ones(env.n_subnetworks) # 
            state, reward, terminated, truncated, info, sum_int = env.step(action)
            rwd[ep, t,:] = reward
    rwd = np.mean(rwd,1)
    return np.mean(rwd), np.max(rwd)


sampler_ = "TPESampler" #"NSGAIISampler"

print("""""""""""""""""""""""""20 episodes min lqr")
print('number of subnetworks = ', n_subnetworks)
print(sampler_)

#search_space = {"x1": np.arange(0, 1, 0.1), "x2": np.arange(0,400,2,dtype=np.int16)}
#sampler = optuna.samplers.GridSampler(search_space=search_space) 
#sampler = optuna.samplers.NSGAIISampler() #
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler = sampler, directions=["minimize","minimize"])
#study = optuna.create_study(sampler = sampler, directions=["minimize"])
study.optimize(objective, n_trials=n_trials)



end = time.time()
print('runtime ', end - start)

print("Number of finished trials: ", len(study.trials))
#optuna.visualization.plot_pareto_front(study)

#study.best_trials  # E.g. {'x': 2.002108042}
print('best trials', study.best_trials)
np.savez(str(n_eps)+str(n_steps)+str(n_trials)+'intlength100_onlyLQR0subband_trials' + str(n_subnetworks) + sampler_ +'.npz',best_trials = study.best_trials, all_trials=study.trials, study=study)
print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

trial_with_lowest_mean_lqr = min(study.best_trials, key=lambda t: t.values[0])
print(f"Trial with lowest mean lqr: ")
print(f"\tnumber: {trial_with_lowest_mean_lqr.number}")
print(f"\tparams: {trial_with_lowest_mean_lqr.params}")
print(f"\tvalues: {trial_with_lowest_mean_lqr.values}")

trial_with_lowest_max_lqr = min(study.best_trials, key=lambda t: t.values[1])
print(f"Trial with lowest mean lqr: ")
print(f"\tnumber: {trial_with_lowest_max_lqr.number}")
print(f"\tparams: {trial_with_lowest_max_lqr.params}")
print(f"\tvalues: {trial_with_lowest_max_lqr.values}")

#print(study.best_trials)
# plt.figure()
#
# optuna.visualization.plot_param_importances(
#     study, target=lambda t: t.values[0], target_name="flops"
# )

#fig = optuna.visualization.plot_contour(study, params=["x", "y"], target=)

# fig = optuna.visualization.plot_pareto_front(study)
# fig.show()
# fig.write_image("fig1.png")
