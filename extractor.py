import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import scipy.special as sp
import torch

def generate_cdf(values, bins_):
    data = np.array(values)
    count, bins_count = np.histogram(data, bins=bins_)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf

# epsilon = 10^(-6)
# n = 1000
# v = 1
# def finite_blocklength_coding(rate, v, n, epsilon):
#     return rate - (np.sqrt(v/n) * sp.erfcinv(2*epsilon)) + (np.log2(n)/(2*n))

mean_lqr = [] #Jointpowerswitch1multi_channel_4115177803030Run1RandomwLQR.npz
method = ['SISA', 'Random', 'useall','RandomwLQR'] #['SISA', 'Random', 'useall'] #'SISAwLQR', 'SISA'  1003030Runnewstep2SISAwLQR.npz multi_channel_intLQR_avAllsumint751958601003030Run1SISAwLQR.npz
context = 'ASswitch_channel_803030Run3' #'multi_channel_intLQR_avAllsumint402701000201003030Run1SISAwLQR.npz' 'multi_channel_intLQR_avAllsumint150400101003030Run1SISAwLQR.npzpz multi_channel_intLQR_avAllsumint40270380601003030Run1SISAwLQR.npz
# file = np.load(context+'SISA'+'.npz', allow_pickle=True)
# rate = file['rate']
# rate = rate[np.nonzero(rate)]
# np.save('Results/sisa_rate', rate)
# x,y = generate_cdf(rate, 1000)
# plt.plot(x,y, label = "channel capacity")
# x,y = generate_cdf(finite_blocklength_coding(rate, v, n, epsilon), 1000)
# plt.plot(x,y, label = "Achievable rate")
# plt.legend()
# plt.grid()
# plt.savefig('Results/sisa_rate.png')


for i in method:
    if i == 'RandomwLQR':
        context = 'ASJointpowerswitch1multi_channel_0.81682378803030Run3'
    file = np.load(context+i+'.npz', allow_pickle=True)
    lqr = file['lqr']
    ul_dl_error = file['ul_dl_error']
    mean_lqr = np.mean(lqr,1)
    #for 
    #ulrate = file['ul_BLER']
    ulrate = file['ulrate']
    #ulrate = np.array(torch.cat(list(ulrate)))
    #ulrate = ulrate[np.nonzero(ulrate)]
    x,y = generate_cdf(ulrate[np.nonzero(ulrate)], 1000)
    plt.plot(x,y, label = i + '_ul')
    
    #dlrate = file['dl_BLER']
    dlrate = file['dlrate']
    # dlrate = np.array(torch.cat(list(dlrate)))
    # dlrate = dlrate[np.nonzero(dlrate)]
    #np.save('Results/sisa_rate', dlrate)
    x,y = generate_cdf(dlrate[np.nonzero(dlrate)], 1000)
    plt.plot(x,y, label = i + '_dl')

    # print(np.argwhere(np.isnan(mean_lqr)).shape)
    # #print(~np.isnan(mean_lqr))
    # mean_lqr = mean_lqr[~np.isnan(mean_lqr)]
    #print(lqr[4,:,76])
    # index = []
    # for j in range(500):
    #     index.append(len(np.where(lqr[j]>4000)[1]))
    # sel = np.argmax(index).astype('int')

    format_99 = "{:.2e}".format(np.percentile(mean_lqr, 99))
    format_999 = "{:.2e}".format(np.percentile(mean_lqr, 99.9))
    format_worst = "{:.2e}".format(np.max(mean_lqr))
    format_mean = "{:.2e}".format(np.mean(mean_lqr))


    print(context + i + ' worst, 99.9th percentile, 99th percentile mean lqr, mean ', [format_worst, format_999, format_99, format_mean])
    print(context + i + ' ul_dl_error_count ',np.sum(ul_dl_error))
#   # np.save('Results/'+context+i+'lqr.npy',lqr[sel])
#     # np.save('Results/'+context+i+'sum_int.npy',file['sum_int'][sel])
#     # np.save('Results/'+context+i+'channel_use.npy',file['channel_use'][sel])
plt.legend()
plt.grid()
plt.savefig('Results/rate_run3.png')