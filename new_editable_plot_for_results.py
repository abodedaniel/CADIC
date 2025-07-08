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

def generate_cdf_(values, num_of_points):
    sorted_data = np.sort(np.array(values).flatten())
    if num_of_points > len(sorted_data):
        num_of_points = len(sorted_data)
        print(len(sorted_data))
    downsample_factor = max(1, len(sorted_data)//num_of_points)
    print(len(sorted_data)//num_of_points)
    #cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    #print('step 1')
    downsampled_data = sorted_data[::downsample_factor]
    #print('step 2')
    downsampled_cdf = np.arange(downsample_factor, len(sorted_data) + downsample_factor, downsample_factor) / len(sorted_data)
    #print('step return')
    return downsampled_data, downsampled_cdf

def generate_cdf(values, num_of_points, metric):
    sorted_data = np.sort(np.array(values).flatten())
    if num_of_points > len(sorted_data):
        num_of_points = len(sorted_data)
    downsample_factor = int(len(sorted_data)/num_of_points)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Retain more points in the tails
    #tail_indices = np.concatenate((np.where(cdf <= 0.05)[0], np.where(cdf >= 0.95)[0]))
    if metric == 'lqr':
        tail_indices = np.where(cdf >= 0.99)[0]
        # Subsample the dense region
        dense_indices = np.where(cdf < 0.99)[0]
    if metric == 'sinr': 
        tail_indices = np.where(cdf >= 1)[0]
        dense_indices = np.where(cdf < 1)[0]
    if metric == 'bler':
        tail_indices = np.where(sorted_data >= 0.500001)[0]
        # Subsample the dense region
        dense_indices = np.where(sorted_data < 0.500001)[0]

    # if metric == 'bler':
    #     tail_indices = np.concatenate((np.where(cdf <= 0.05)[0], np.where(cdf >= 0.95)[0]))
    #     dense_indices = np.where((cdf > 0.05) & (cdf < 0.95))[0]

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


numsubn = 15


run = [6,7,8]

metric = 'bler'
method_ = 'SISA+PC'

algo = {
    #'Random N=10': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + 'Random.npz' for i in run],
    # 'SISA N=10': ['Mixedplant3RBswitch_channel_103030Run' +str(i) + method_ + '.npz' for i in run],
    #'Random N=15': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'Random.npz' for i in run],
    #'Random N=15 10Ghz': ['wTcounterw10GHzprodMixedplant3RBswitch_channel_153030Run' +str(i) + 'Random.npz' for i in [7,8]],
    #'SISA N=15': ['wTcounterprodMixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA.npz' for i in [8]],
    #'Everytime SISA N=15': ['EverytimestepMixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA.npz' for i in run],
    # 'SISA N=15': ['Mixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA' + '.npz' for i in [6,7]],
    # 'SISA N =30': ['MixedPlantswitch_channel_303030Run6SISA.npz']
    #'SISA+PC N =15': ['minMixedplant3RBswitch_channel_153030Run' +str(i) + method_ + '.npz' for i in [7,8,9]],
    'SISA+PC N =15': ['wTcounterw10GHzprodMixedplant3RBswitch_channel_153030Run' +str(i) + method_ + '.npz' for i in [7]], 
    #'prodSISA+PC N =15': ['prodMixedplant3RBswitch_channel_153030Run' +str(i) + 'SISA+PC' + '.npz' for i in run],
    #'Useall': ['Mixedplant3RBswitch_channel_'+str(numsubn)+'3030Run' +str(i) + 'useall.npz' for i in [6]],
    #'CAIC N=10': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186103030Run'+str(i)+'RandomwLQR.npz' for i in run],
    #'CAIC N=15': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run'+str(i)+'RandomwLQR.npz' for i in [6,7]],
    #'CAIC N=15 wTC zerod -20dBm': ['wTcounterMixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run'+str(i)+'RandomwLQR.npz' for i in [7,8]],
    #'CAIC N=15 wTC zerod -15dBm': ['wTcounterzerod15dBmMixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run7RandomwLQR.npz', 'wTcounterzerod15dBmMixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186203030Run8RandomwLQR.npz'],
    #'CAIC N=15 wTC zerod -25dBm': ['wTcounterzerod25dBmMixedplant3RBJointpowerswitch1multi_channel_TestN150.4916100186153030Run'+str(i)+'RandomwLQR.npz' for i in [6,7]],
    #'CAIC N=20': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN100.642096146203030Run'+str(i)+'RandomwLQR.npz' for i in run],
    #'CAIC N=10, 0.4': ['Mixedplant3RBJointpowerswitch1multi_channel_TestN100.42096146103030Run'+str(i)+'RandomwLQR.npz' for i in run],
    # 'CICA-PC N=15': ['wTcounterMixedplant3RBpowerswitch0.8536153030Run' +str(i) + 'CICA-PC.npz' for i in run],
    # 'CICA-PC N=15 0.23': ['wTcounterMixedplant3RBpowerswitch0.236153030Run' +str(i) + 'CICA-PC.npz' for i in [8]],
}



i = 1
state_x_ = []
labels= []
force_ = []
lqr_all_ = []

lqr_percentile = {}
for method in algo:
    print(method)
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
            try:
                print(method, file['transmission_counter'])
            except:
                continue

        if metric == 'sinr': 
            ul_sinr = file['ul_sinr']
            ul_sinr_.extend(list(ul_sinr.flatten()))

        # dl_sinr = file['dl_sinr']
        # dl_sinr_.extend(list(dl_sinr.flatten()))
        if metric == 'bler': 
            ul_bler = file['dl_BLER']
            ul_bler_.extend(list(ul_bler.flatten()))

            # dl_bler = file['dl_BLER']
            # dl_bler_.extend(list(dl_bler.flatten()))

    #np.savez()
    if metric == 'lqr': 
        x,y = generate_cdf(mean_lqr, 500, metric)
        plt.loglog(x, 1-y, label=method)
        print('step plot')
        plt.title('Mean LQR')
        plt.ylabel('CCDF')
        plt.xlabel('Mean LQR')
        #plt.legend()
        plt.grid()
        #ax1.set_xlim([10**0,10**5])
        plt.ylim([10**-4,10**0])
        plt.xlim([10**0,1e3])
        print(x.shape)
    if metric == 'sinr': 
        x,y = generate_cdf(ul_sinr_, 500, metric)
        plt.semilogy(10*np.log10(x), y, label=method+'UL')
        # x,y = generate_cdf(dl_sinr_, 100)
        # plt.semilogx(x, y) #,'-.' , label=method+'DL')
        plt.title('SINR (dB)')
        plt.ylabel('CDF')
        plt.xlabel('SINR (dB)')
        plt.legend()
        plt.grid()
        print(x.shape)
    if metric == 'bler':
        x,y = generate_cdf_(ul_bler_, 50000)
        plt.loglog(x[x > 10**-5], 1-y[x > 10**-5], label=method+'UL')
        # x,y = generate_cdf_(dl_bler_, 100000)
        # plt.loglog(x[x > 10**-5], 1-y[x > 10**-5]) #, '-.' ,label=method+'DL')
        plt.title('Block error rate')
        plt.ylabel('CDF')
        plt.xlabel('BLER')
        #plt.legend()
        plt.grid()
        plt.xlim([0,1])
        plt.ylim([10**-5,10**0])
        print(x[x > 10**-5].shape)


#plt.savefig('Results/2Mixedplant_dl'+method_+metric+'result.png')
tikzplotlib.save('Results/Mixedplantdl'+method_+metric+'result.tex')
#np.save('BufferweightedSR meanLQR.npy',lqr_percentile)