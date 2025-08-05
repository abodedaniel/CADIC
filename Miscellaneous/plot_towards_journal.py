import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import tikzplotlib
np.set_printoptions(precision=4, suppress=True)
#import seaborn as sns

# def generate_cdf(values, bins_):
#     data = np.array(values)
#     count, bins_count = np.histogram(data, bins=bins_)
#     pdf = count / sum(count)
#     cdf = np.cumsum(pdf)
#     return bins_count[1:], cdf
# def generate_cdf(values, num_of_points):
#     sorted_data = np.sort(np.array(values).flatten())
#     if num_of_points > len(sorted_data):
#         num_of_points = len(sorted_data)
#     downsample_factor = int(len(sorted_data)/num_of_points)
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#     downsampled_data = sorted_data[::downsample_factor]
#     downsampled_cdf = cdf[::downsample_factor]
#     return downsampled_data, downsampled_cdf

def generate_cdf(values, num_of_points):
    sorted_data = np.sort(np.array(values).flatten())
    if num_of_points > len(sorted_data):
        num_of_points = len(sorted_data)
    downsample_factor = int(len(sorted_data)/num_of_points)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Retain more points in the tails
    tail_indices = np.concatenate((np.where(cdf <= 0.05)[0], np.where(cdf >= 0.95)[0]))

    # Subsample the dense region
    dense_indices = np.where((cdf > 0.05) & (cdf < 0.95))[0]
    sparse_dense_indices = dense_indices[::downsample_factor]

    # Combine tail and sparse dense indices
    selected_indices = np.unique(np.concatenate((tail_indices, sparse_dense_indices)))

    downsampled_data = sorted_data[selected_indices]
    downsampled_cdf = cdf[selected_indices]
    return downsampled_data, downsampled_cdf


def findcdfvalue(x,y,yval1,yval2):
    a = x[np.logical_and(y>yval1, y<yval2)]
    if a.size < 1:
        return 0
    else:
        m = np.mean(a)
        return m.item()

def process_info(file_object):
    lqr = file_object['lqr']
    plant_monitor = file_object['plant_monitor']
    count_all_failed = 0
    Availability_all = []
    Availability_failed = []
    lqr_notfailed = np.zeros_like(lqr)
    for i in range(200):
        failed = np.any(plant_monitor[i,:,:], axis=0)
        for n in range(plant_monitor.shape[-1]):
            Availability_all.append(np.sum(plant_monitor[i,:,n]==0)/plant_monitor[i,:,n].shape[0])
            if failed[n]:
                Availability_failed.append(np.sum(plant_monitor[i,:,n]==0)/plant_monitor[i,:,n].shape[0])
        count_all_failed += np.sum(failed)
        lqr_notfailed[i,:,0:lqr[i,:,:][:,~failed].shape[1]] = lqr[i,:,:][:,~failed]
    return lqr_notfailed, count_all_failed, Availability_all, Availability_failed


numsubn = 10

# algo = {'ASswitch_channel_'+str(numsubn)+'3030Run3useall.npz':'Fixed Power',
#         'ASswitch_channel_'+str(numsubn)+'3030Run3Random.npz':'Random SSA',
#         #'initASswitch_channel_'+str(numsubn)+'3030Run3Random.npz':'Random SSA (Every episode init)',
#         'ASswitch_channel_'+str(numsubn)+'3030Run3SISA.npz':'SISA (Episode init)',
#         'ASswitch_channel_'+str(numsubn)+'3030Run3superRandom.npz':'superRandom',
#         'ASJointpowerswitch1multi_channel_0.81682378'+str(numsubn)+'3030Run3RandomwLQR.npz':'CAIC S1=82,S2=378, K=0.8, X0=16'
# }

# algo = {
#         'Plant005TS1switch_channel_'+str(numsubn)+'3030Run6SISA.npz':'SISA',
#         'Plant005TS1switch_channel_'+str(numsubn)+'3030Run6Random.npz':'Random SSA',
#         'Plant005TS1Jointpowerswitch1multi_channel_0.3562849296'+str(numsubn)+'3030Run6SISAwLQR.npz':'CAIC SISA S1=92,S2=96, K=0.356, X0=284',
#         'Plant005TS1Jointpowerswitch1multi_channel_0.110092400'+str(numsubn)+'3030Run6SISAwLQR.npz':'CAIC SISA S1=92,S2=400, K=0.1, X0=100',
#         'Plant005TS1Jointpowerswitch1multi_channel_0.3562849296'+str(numsubn)+'3030Run6RandomwLQR.npz':'CAIC Random S1=92,S2=96, K=0.356, X0=284',
#         'Plant005TS1Jointpowerswitch1multi_channel_0.03284200400'+str(numsubn)+'3030Run6RandomwLQR.npz': 'CAIC Random S1=200,S2=400, K=0.03, X0=284',
#         'Plant005TS1Jointpowerswitch1multi_channel_0.03284200400'+str(numsubn)+'3030Run6SISAwLQR.npz': 'CAIC SISA S1=200,S2=400, K=0.03, X0=284',
# }

# algo = {
#         'MixedPlantswitch_channel_'+str(numsubn)+'3030Run6SISA.npz':'SISA',
#         'MixedPlantswitch_channel_'+str(numsubn)+'3030Run6Random.npz':'Random SSA',
#         'MixedPlantJointpowerswitch1multi_channel_0.03284200400'+str(numsubn)+'3030Run6RandomwLQR.npz': 'CAIC Random S1=200,S2=400, K=0.03, X0=284',
# }


# algo = {
#         'ASswitch_channel_'+str(numsubn)+'3030Run5SISA.npz':'SISA',
#         'ASswitch_channel_'+str(numsubn)+'3030Run5Random.npz':'Random SSA',
#         'Plant005TS2Jointpowerswitch1multi_channel_0284200400'+str(numsubn)+'3030Run6RandomwLQR.npz': 'CAIC Random S1=200,S2=400, K=0, X0=284',
#         'Plant005TS2Jointpowerswitch1multi_channel_0.03284200400'+str(numsubn)+'3030Run6RandomwLQR.npz': 'CAIC Random S1=200,S2=400, K=0.03, X0=284',

# }
run = [6,7,8]
# algo = {
#         'intermediateMixedplant3RBswitch_channel_'+str(numsubn)+'3030Run6SISA.npz':'SISA',
#         'intermediateMixedplant3RBswitch_channel_'+str(numsubn)+'3030Run6Random.npz':'Random',
#         'Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run8SISA.npz': 'SISA 2',
#         'Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run8Random.npz': 'Random 2',
#         'Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run8useall.npz': 'Useall 2',
#         'Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run7SISA.npz': 'SISA 3',
#         'Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run7Random.npz': 'Random 3',
#         'Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run7useall.npz': 'Useall 3',
#         #'intermediateMixedplant3RBswitch_channel_'+str(numsubn)+'3030Run6useall.npz':'Useall',
#         # 'Plant005TS2Jointpowerswitch1multi_channel_0284200400'+str(numsubn)+'3030Run6RandomwLQR.npz': 'CAIC Random S1=200,S2=400, K=0, X0=284',
#         # 'Plant005TS2Jointpowerswitch1multi_channel_0.03284200400'+str(numsubn)+'3030Run6RandomwLQR.npz': 'CAIC Random S1=200,S2=400, K=0.03, X0=284',
# }

algo = {
    'Random N=10': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + 'Random.npz' for i in run],
    'SISA N=10': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + 'SISA.npz' for i in run],
    'Random N=15': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'Random.npz' for i in run],
    'SISA N=15': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA.npz' for i in run],
    'SISA N=20': ['Mixedplant3RBswitch_channel_203030Run' +str(i) + 'SISA.npz' for i in run],
    #'Useall': ['Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run' +str(i) + 'useall.npz' for i in run],
}



fig = plt.figure(figsize=[7,15])

#fig_ = fig.subfigures(2, wspace=0.2, height_ratios=[2,6])

#(ax1) = fig_[0].subplots(1)
#fig_[0].subplots_adjust(wspace=0.05, hspace=0)


(ax1, ax2, ax3) = fig.subplots(3) 
#fig_[1].subplots_adjust(wspace=0.5, hspace=0.2)
#directory = 'Results202020'
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
        lqr = file['lqr']
        mean_lqr = np.mean(lqr,1)
        mean_lqr_.extend(list(mean_lqr.flatten()))

        ul_sinr = file['ul_sinr']
        ul_sinr_.extend(list(ul_sinr.flatten()))

        dl_sinr = file['dl_sinr']
        dl_sinr_.extend(list(dl_sinr.flatten()))

        ul_bler = file['ul_BLER']
        ul_bler_.extend(list(ul_bler.flatten()))

        dl_bler = file['dl_BLER']
        dl_bler_.extend(list(dl_bler.flatten()))



    x,y = generate_cdf(mean_lqr_, 20000)
    ax1.loglog(x, 1-y) #, label=method)

    x,y = generate_cdf(ul_sinr_, 20000)
    ax2.semilogx(x, y) #, label=method+'UL')
    x,y = generate_cdf(dl_sinr_, 20000)
    ax2.semilogx(x, y) #,'-.' , label=method+'DL')

    x,y = generate_cdf(ul_bler_, 20000)
    ax3.loglog(x, y) #, label=method+'UL')
    x,y = generate_cdf(dl_bler_, 20000)
    ax3.loglog(x, y) #, '-.' ,label=method+'DL')





ax1.set_title('Mean LQR')
ax1.set_ylabel('CCDF')
ax1.set_xlabel('Mean LQR')
#ax1.legend()
ax1.grid()
#ax1.set_xlim([10**0,10**5])
ax1.set_ylim([10**-4,10**0])
ax1.set_xlim([10**0,1e3])
#ax1.set_ylim([0.99,1.0])

ax2.set_title('SINR (dB)')
ax2.set_ylabel('CDF')
ax2.set_xlabel('SINR (dB)')
#ax2.legend()
ax2.grid()
#ax2.set_xlim([10**-0.8,100])
#ax2.set_ylim([10**-3,1.0])

ax3.set_title('Block error rate')
ax3.set_ylabel('CDF')
ax3.set_xlabel('BLER')
#ax3.legend()
ax3.grid()
ax3.set_xlim([10**-5,1])
ax3.set_ylim([10**-1,1.0])



# ax14.set_ylabel('Instantaneous Pole angle (rad)')

# ax9.grid()
# ax9.set_xticks([0.1,0.2,0.4,0.5,0.8,1,1.5,2])
# ax9.set_xticklabels(['0.1','0.2','0.4','0.5','0.8','1','1.5','2'])
# #ax9.legend() #fontsize='x-small', loc='upper right')
# ax9.set_xlim([0.1,2])
# ax9.set_ylim([10**-3,1.0])


# ax14.grid()
# ax14.set_xticks(list(np.arange(0,1.1,0.1)*np.pi))
# ax14.set_xticklabels([0, '0.1$\pi$','0.2$\pi$','0.3$\pi$','0.4$\pi$','0.5$\pi$','0.6$\pi$','0.7$\pi$','0.8$\pi$','0.9$\pi$','1.0$\pi$'])
# #ax14.legend() #(fontsize='x-small', loc='upper right')
# ax14.set_xlim([0,np.pi])
# ax14.set_ylim([10**-3,1.0])


#fig.tight_layout()

#tikzplotlib.save(str(numsubn)+'State_error.tex')
#fig.savefig('1x'+str(numsubn)+'newResults.png')
#np.save(str(numsubn)+'99percentilemeanLQR.npy',lqr_percentile)

tikzplotlib.save('Results/Mixedplantresult.tex')
#fig.savefig('Results/newnew3RBintermediateresult'+str(numsubn)+'MixedplantInit_result.png')
#np.save('BufferweightedSR meanLQR.npy',lqr_percentile)