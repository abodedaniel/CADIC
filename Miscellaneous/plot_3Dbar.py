import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import scipy.special as sp
import torch
import tikzplotlib


numsubn = [10,15,20]


run = [6,7,8]

metric = 'lqr'
method_ = 'SISA+PC'
methods = {'CADIC-P':[5,5,5,5,5,5], 'CADIC-M':[5,5,5,5,5,5], 'SISA': [5,5,5,5,5,5], 'SISA+PC':[5,5,5,5,5,5], 'Random':[5,5,5,5,5,5]}

algo = {
        20:{'CADIC-P':['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186203030Run'+str(i)+'RandomwLQR.npz' for i in run],
            'CADIC-M':['wTcounterzerod25Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186203030Run' +str(i) + 'RandomwLQR.npz' for i in run],
            'SISA': ['Mixedplant3RBswitch_channel_203030Run' +str(i) + 'SISA.npz' for i in run],
            'SISA+PC': ['prodMixedplant3RBswitch_channel_203030Run' +str(i) + method_ + '.npz' for i in run],
            'Random': ['Mixedplant3RBswitch_channel_203030Run' +str(i) + 'Random.npz' for i in run],
            },
        15:{'CADIC-P':['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run'+str(i)+'RandomwLQR.npz' for i in run],
            'CADIC-M':['wTcounterzerod25dBmMixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run' +str(i) + 'RandomwLQR.npz' for i in run],
            'SISA': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA.npz' for i in run],
            'SISA+PC': ['prodMixedplant3RBswitch_channel_153030Run' +str(i) + method_ + '.npz' for i in run],
            'Random': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'Random.npz' for i in run],
            },
        10:{'CADIC-P':['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186103030Run'+str(i)+'RandomwLQR.npz' for i in run],
            'CADIC-M':['wTcounterzerod25Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186103030Run' +str(i) + 'RandomwLQR.npz' for i in run],
            'SISA': ['2Mixedplant3RBswitch_channel_103030Run' +str(i) + 'SISA.npz' for i in run],
            'SISA+PC': ['prodMixedplant3RBswitch_channel_103030Run' +str(i) + method_ + '.npz' for i in run],
            'Random': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + 'Random.npz' for i in run],
            }
            }




i = 1
state_x_ = []
labels= []
force_ = []
lqr_all_ = []
l_99 = [3,4,5]
l_99_9 = [0,1,2]
lqr_percentile = {}
i = 0
for num in algo:
    for method in algo[num]:
        mean_lqr_ = []
        #file = filename #os.path.join(directory, filename)
        filenames = algo[num][method]
        for filename in filenames:
            try:
                file = np.load(filename, allow_pickle=True)
                lqr = file['lqr']
                mean_lqr = np.mean(lqr,1)
                mean_lqr_.extend(list(mean_lqr.flatten()))
            except Exception as e:
                continue
        try:
            format_99 = "{:.2e}".format(np.percentile(mean_lqr_, 99))
            format_999 = "{:.2e}".format(np.percentile(mean_lqr_, 99.9))
            format_worst = "{:.2e}".format(np.max(mean_lqr_))
            format_mean = "{:.2e}".format(np.mean(mean_lqr_))
            #print(num, method, l_99[i], l_99_9[i], methods[method][l_99[i]])
            methods[method][l_99[i]] = np.percentile(mean_lqr_, 99)
            methods[method][l_99_9[i]] = np.percentile(mean_lqr_, 99.9)
        except IndexError as e:
            format_99 = 10
            format_999 = 10

        print(str(num) + 'N _'+ method + ' 99th percentile, 99.9th percentile', [format_99, format_999])
    i = i+1

print(methods)
np.savez('methods_lqr_bars',methods)
plt.rcParams["figure.figsize"] = [8.00, 4.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure(layout='tight')
ax1 = fig.add_subplot(111, projection='3d')
x = np.arange(1,18,3)
x_ticks = ['N=20, 99.9%', 'N=15, 99.9%', 'N=10, 99.9%', 'N=20, 99%', 'N=15, 99%', 'N=10, 99%']
y = np.repeat(np.expand_dims(np.arange(1,18,3),-1), 6,axis=1)
dx = 2*np.ones(6)
dy = 2* np.ones(6)
z3 = np.ones(6)

colors = ['yellow','blue', 'green', 'red','cyan','purple']
i = 0
for method in methods:
    ax1.bar3d(x,y[i],z3, dx, dy, methods[method], color=colors[i])
    i = i+1

ax1.set_xticks(x+1)
ax1.set_xticklabels(x_ticks, rotation=-150, ha="left", rotation_mode="anchor")
ax1.set_yticks(np.arange(3,17,3)-0.8)
ax1.set_yticklabels(list(methods), rotation=-90, ha="center")
ax1.set_zscale('log')
ax1.view_init(elev=10, azim=-30)
fig.set_tight_layout(True)
#ax1.legend([b1, b2], ['1', '2'])
plt.savefig('Results/lqrbarplot.png')
tikzplotlib.save('Results/lqrbarplot.tex')


