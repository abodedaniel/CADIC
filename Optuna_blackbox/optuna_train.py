import optuna
import matplotlib.pyplot as plt
from logistic_lqr import logistic_agent
from control_aware_subn_env import SubnetworkEnv
import numpy as np
import time

start = time.time()
def objective(trial):
    x1 = trial.suggest_float('x1', 0, 1, step=0.01)
    x2 = trial.suggest_int('x2', 0, 200, step=2)

    n_eps = 10
    n_steps = 1000
    n_subnetworks = 80

    env = SubnetworkEnv(num_step=n_steps, n_subnetworks=n_subnetworks)       
    rwd = np.zeros((n_eps, n_steps,env.n_subnetworks))      
    agent = logistic_agent(n_subn=env.n_subnetworks, Pmax=0)

    for ep in range(n_eps):  # Run episodes
        #env.seed(seed=np.random.randint(999999999))
        reward = 0
        done = False
        state, info = env.reset()
        for t in range(n_steps):  # Run one episode
            action = agent.act(state,[x1,x2]) #np.ones(env.n_subnetworks) # 
            state, reward, terminated, truncated, info = env.step(action)
            rwd[ep, t,:] = reward
    rwd = np.mean(rwd,1)
    return np.max(rwd) #np.mean(rwd), 


n_subnetworks = 80
sampler_ = "TPESampler" #"NSGAIISampler"

print("""""""""""""""""""""""""20 episodes min lqr")
print('number of subnetworks = ', n_subnetworks)
print(sampler_)

search_space = {"x1": np.arange(0, 1, 0.01), "x2": np.arange(0,200,2,dtype=np.int16)}
#sampler = optuna.samplers.GridSampler(search_space=search_space) 
#sampler = optuna.samplers.NSGAIISampler() #
sampler = optuna.samplers.TPESampler()
#study = optuna.create_study(sampler = sampler, directions=["minimize","minimize"])
study = optuna.create_study(sampler = sampler, directions=["minimize"])
study.optimize(objective, n_trials=400)



end = time.time()
print('runtime ', end - start)

print("Number of finished trials: ", len(study.trials))
#optuna.visualization.plot_pareto_front(study)

#study.best_trials  # E.g. {'x': 2.002108042}
print('best trials', study.best_trials)
np.savez('maxlqrpower_trials' + str(n_subnetworks) + sampler_ +'.npz',best_trials = study.best_trials, all_trials=study.trials, study=study)
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
