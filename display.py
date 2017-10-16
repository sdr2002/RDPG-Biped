__author__ = 's1687487'

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# perf_dict_name = os.environ.get('PERF_DICT')
# perf_dict = pickle.load(file=open(name=perf_dict_name,mode='rb'))
perf_dict = pickle.load(file=open(name='perf.p',mode='rb'))

fig, ax1 = plt.subplots()
#=============== plotting
try:
    len_info = len(perf_dict['R_tr_real'])
    if len_info == 0:
        raise
except Exception as e:
    len_info = 1
    perf_dict['R_tr_real'] = np.array([])
R_tr_real = np.append(perf_dict['R_tr'][:-len_info],perf_dict['R_tr_real'][-len_info:])

ln1 = ax1.plot(R_tr_real, label='R_tr', linestyle = '-', color='r')
R_tr_ma = moving_average(R_tr_real)
ln2 = ax1.plot(R_tr_ma, label='R_tr(100MA)', linestyle = '-', color='orange')

ax1.set_ylabel('Acc. Reward')
ax1.set_ylim([-50.,400.])
ax2=ax1.twinx()

print('Length of dP-C:'+str(len(perf_dict['dParam_Critic'])))
#ln3 = ax2.plot(perf_dict['dParam_Critic'], label='dParam_Critic', linestyle = '-.', color='g', alpha=1.)
ln4 = ax2.plot(perf_dict['dParam_Actor'], label='dParam_Actor', linestyle = '-.', color='b', alpha=1.)
#===============
# print diff_avg_list
ax2.set_ylabel('Norm of change')
ax2.set_yscale('log')

#if not len(perf_dict['R_te']) == 0:
#lns = ln1 + ln2+ ln3 + ln4
#else:
lns = ln1 + ln2+ ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=4)

# plt.savefig(DIRECTORY+'/results/EpiLog',format='pdf')
plt.show()
