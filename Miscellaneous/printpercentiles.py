import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import scipy.special as sp
import torch


numsubn = 15


run = [6,7,8]

metric = 'lqr'
method_ = 'CADIC'

algo = {
    #'Random N=10': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + 'Random.npz' for i in run],
    # 'SISA N=10': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + method_ + '.npz' for i in run],
    #'Random N=15': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'Random.npz' for i in run],
    #'SISA N=15': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA.npz' for i in run],
    # 'SISA N=20': ['Mixedplant3RBswitch_channel_203030Run' +str(i) + method_ + '.npz' for i in run],
    # 'SISA N =30': ['MixedPlantswitch_channel_303030Run6SISA.npz']
    #'SISA+PC N =15': ['minMixedplant3RBswitch_channel_153030Run' +str(i) + method_ + '.npz' for i in [7,8,9]],
    #'SISA+PC N =15': ['prodMixedplant3RBswitch_channel_153030Run' +str(i) + method_ + '.npz' for i in run],
    #'Useall': ['Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run' +str(i) + 'useall.npz' for i in [9]],
    #'CAIC N=10': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186103030Run'+str(i)+'RandomwLQR.npz' for i in run],
    #'CAIC N=15': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run'+str(i)+'RandomwLQR.npz' for i in [6,7]],
    #'CAIC N=20': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN100.642096146203030Run'+str(i)+'RandomwLQR.npz' for i in run],
    #'CAIC N=10, 0.4': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN100.42096146103030Run'+str(i)+'RandomwLQR.npz' for i in run],
    #'CADIC Train 20 test 20' : ['counterzerod180Mixedplant3RBJointpowerswitch1multi_channel_TestN200.6614130246203030Run'+ str(i) + 'RandomwLQR.npz' for i in [6,7]],
    #'CADIC Train 20 test 15' : ['counterzerod180Mixedplant3RBJointpowerswitch1multi_channel_TestN200.6614130246153030Run'+ str(i) + 'RandomwLQR.npz' for i in [7,8]],
    #'CADIC Train 20 test 10' : ['counterzerod180Mixedplant3RBJointpowerswitch1multi_channel_TestN200.6614130246103030Run'+ str(i) + 'RandomwLQR.npz' for i in [6,7]],
    'CADIC Train 10 test 15' : ['counterzerod180Mixedplant3RBJointpowerswitch1multi_channel_TestN100.42096146153030Run'+ str(i) + 'RandomwLQR.npz' for i in [6,7]],  
    #'CADIC Train 10 test 10' : ['Mixedplant3RBJointpowerswitch1multi_channel_TestN100.42096146103030Run'+ str(i) + 'RandomwLQR.npz' for i in [6,7,8]],
    'CADIC Train 10 test 20' : ['counterzerod180Mixedplant3RBJointpowerswitch1multi_channel_TestN100.42096146203030Run'+ str(i) + 'RandomwLQR.npz' for i in [6,7]],
}


i = 1
state_x_ = []
labels= []
force_ = []
lqr_all_ = []

lqr_percentile = {}
for method in algo:
    mean_lqr_ = []
    ul_sinr_ = []
    dl_sinr_ = []
    dl_bler_ = []
    ul_bler_ =[]
    #file = filename #os.path.join(directory, filename)
    filenames = algo[method]
    for filename in filenames:
        file = np.load(filename, allow_pickle=True)

        if metric == 'lqr': 
            lqr = file['lqr']
            mean_lqr = np.mean(lqr,1)
            mean_lqr_.extend(list(mean_lqr.flatten()))

    # print(np.argwhere(np.isnan(mean_lqr)).shape)
    # #print(~np.isnan(mean_lqr))
    # mean_lqr = mean_lqr[~np.isnan(mean_lqr)]
    #print(lqr[4,:,76])
    # index = []
    # for j in range(500):
    #     index.append(len(np.where(lqr[j]>4000)[1]))
    # sel = np.argmax(index).astype('int')

    #format_99 = "{:.2e}".format(np.percentile(mean_lqr, 99))
    format_999 = "{:.2e}".format(np.percentile(mean_lqr, 99.9))
    format_worst = "{:.2e}".format(np.max(mean_lqr))
    format_mean = "{:.2e}".format(np.mean(mean_lqr))


    print(method + '99.9th percentile', [format_999])

