import optuna
import matplotlib.pyplot as plt
from piecewise_lqr_step import piecewise_agent
from control_aware_subn_joint_power_subband_env import SubnetworkEnv
import numpy as np
import time

global n_subnetworks, n_eps, n_steps, n_subband
n_eps = 100
n_steps = 1000
n_subnetworks = 15
n_subband = 3
n_trials = 400
n_startup_trials = 100

start = time.time()
def objective(trial):
    K = trial.suggest_float('K', 0, 1, step=0.01)
    X0 = trial.suggest_int('X0', 0, 150, step=1)
    # S2 = trial.suggest_int('S2', X0, 100, step=2)
    # S3 = trial.suggest_int('S3',S2, 150, step=2)
    #x3 = trial.suggest_int('x3', 0, 400, step=5)
    
    env = SubnetworkEnv(num_step=n_steps, n_subnetworks=n_subnetworks, n_subband = n_subband)       
    rwd = np.zeros((n_eps, n_steps,env.n_subnetworks))      
    agent = piecewise_agent(n_subn=env.n_subnetworks, num_subband= n_subband, Pmax=0)

    for ep in range(n_eps):  # Run episodes
        #env.seed(seed=np.random.randint(999999999))
        reward = 0
        done = False
        state, info, sum_int, _ = env.reset(options='Random')
        #print(sum_int)
        for t in range(n_steps):  # Run one episode
            pow_action = agent.power_act(state,[K,X0]) #np.ones(env.n_subnetworks) # 
            ch_action = np.ones((env.n_subnetworks, n_subband)) * np.array([[0,1,2]])
            state, reward, terminated, truncated, info, sum_int = env.step(pow_action, ch_action)
            env.init_ch = ch_action
            rwd[ep, t,:] = reward
        # intermediate_value = np.mean(rwd,1)
        # intermediate_value = np.mean(intermediate_value)
        # trial.report(intermediate_value, ep)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
    rwd = np.mean(rwd,1)
    return np.mean(rwd), np.max(rwd)


sampler_ = "TPESampler" #"NSGAIISampler"

print("""""""""""""""""""""""""20 episodes min lqr")
print('number of subnetworks = ', n_subnetworks)
print(sampler_)

#search_space = {"x1": np.arange(0, 400, 5,dtype=np.int16), "x2": np.arange(0,400,5,dtype=np.int16), "x3": np.arange(0,400,5,dtype=np.int16)}
#sampler = optuna.samplers.GridSampler(search_space=search_space) 
#sampler = optuna.samplers.NSGAIISampler() #
sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=100)
#sampler = optuna.samplers.GPSampler(seed=0, n_startup_trials=100)
study = optuna.create_study(sampler = sampler, directions=["minimize","minimize"])
#study = optuna.create_study(sampler = sampler, directions=["minimize"])#, pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=n_trials)



end = time.time()
print('runtime ', end - start)

print("Number of finished trials: ", len(study.trials))
#optuna.visualization.plot_pareto_front(study)

#study.best_trials  # E.g. {'x': 2.002108042}
print('best trials', study.best_trials)
np.savez('MixedplantsOnlypower150Random'+str(n_eps)+str(n_steps)+str(n_trials)+'subband_trials' + str(n_subnetworks) + sampler_ +'.npz',best_trials = study.best_trials, all_trials=study.trials, study=study)
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
