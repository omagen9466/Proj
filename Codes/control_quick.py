import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import main 
import dropwise
import hom
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from pylab import *

#Following region D2 Dropwise Compare
#region

# Q = [0]
# # Q=[2e-4]
# Temp_diff=[1]
# # df_main = pd.DataFrame()
# # df_condense= pd.DataFrame()
# # for i in Q:
# dropwise_step=1e-14 
# end=5e-8
# dT=10
# dict=dropwise.Condense(T_wall=373-dT,theta=10,endtime=end,step=dropwise_step)
# dict1 = main.D2(Q_ext=dict['q'],T_s=372-dT,delta_t=dropwise_step,r_s0=dict['r_min']*1.001,endtime=end)
      
# print('Finished creating DF')


# sns.set_style("whitegrid")


# fig_k = plt.figure(figsize=(10,5))
# ax = fig_k.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['k_slope'][0],label='Dropwise')
# ax.plot(dict1['time'],dict1['k_slope'][0],label='$D^{2}$')
# ax.set_xlabel('Time [$s$]')
# ax.set_ylabel('$k$ [$m/s$]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# # fig_k.savefig('k.pdf')

# fig_r = plt.figure(figsize=(10,5))
# ax = fig_r.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['r'][0],label='Dropwise')
# ax.plot(dict1['time'],dict1['rs'],label='$D^{2}$')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('r [m]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# # fig_r.savefig('r.pdf')

# fig_k_sq = plt.figure(figsize=(10,5))
# ax = fig_k_sq.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['k_sq'][0],label='Dropwise')
# ax.plot(dict1['time'],dict1['k_sq'][0],label='$D^{2}$')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$k^{2}$ [$m^{2}/s^{2}$]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# # fig_k_sq.savefig('k_sq.pdf')

# fig_r_sq = plt.figure(figsize=(10,5))
# ax = fig_r_sq.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['r_sq'][0][0],label='Dropwise')
# ax.plot(dict1['time'],dict1['r_sq'],label='$D^{2}$')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$r^{2}$ [$m^{2}$]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# # fig_r_sq.savefig('r_sq.pdf')
# plt.show()
# endregion

#Following is D2 Knudsen Compare
#region

# save=False
# step=1e-8
# endtime=1e-4
# fig_k = plt.figure(figsize=(12,10))
# fig_k_sq = plt.figure(figsize=(12,10))
# fig_r_sq_lin = plt.figure(figsize=(12,10))
# dT=[1,10]
# for i in range(1,3):
#     dict=hom.Knudsen(T_s=373+dT[i-1],T_inf=373,step=step,endtime=endtime,r0=1e-6)
#     dict1 = main.D2(Q_ext=0,T_s=350+dT[i-1],T_inf=350,delta_t=step,r_s0=1e-6,endtime=endtime,cond=False)
#     dict2 = main.D2(Q_ext=0,T_s=350+dT[i-1],T_inf=350,delta_t=step,r_s0=1e-6,endtime=endtime,cond=True)  
#     print('Finished creating DF')


#     y_formatter = ScalarFormatter(useOffset=False)


#     sns.set_style("whitegrid")
#     plt.rc('font', size=17) 
    
#     ax = fig_k.add_subplot(2, 1, i)
#     ax.set_title('d$T$={}K'.format(dT[i-1]),fontsize=17)
#     # ax.plot(dict['time'][0],dict['k_slope'][0],label='Knudsen')
#     ax.plot(dict1['time'],dict1['T_s'],label='$d^{2}$: $T_{s}$=Const')
#     ax.plot(dict2['time'],dict2['T_s'],label='$d^{2}$: Infinite Conductivity',color='C2')
#     ax.yaxis.set_major_formatter(y_formatter)
#     if i==2:
#         ax.set_xlabel('Time [s]')
#     else:
#         ax.legend()
#     ax.set_ylabel('$T_{s}$ [K]')

#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
    
    
#     if save:
#         fig_k.savefig('../Reports/Final_report/Figures/D2_dT.pdf')


#     # fig_k = plt.figure(figsize=(12,5))
#     # ax = fig_k.add_subplot(1, 1, 1)
#     # ax.plot(dict['time'][0],dict['k_slope'][0],label='Knudsen')
#     # ax.plot(dict1['time'],dict1['k_slope'][0],label='$d^{2}$: $T_{s}$=Const')
#     # ax.plot(dict2['time'],dict2['k_slope'][0],label='$d^{2}$: Infinite Conductivity')
#     # ax.set_xlabel('Time [$s$]')
#     # ax.set_ylabel('$k_{diff}$ [$m/s$]')
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
#     # ax.legend()
#     # if save:
#     #     fig_k.savefig('../Reports/Final_report/Figures/k_hom_vs_D2_1dT.pdf')

#     fig_r = plt.figure(figsize=(12,5))
#     ax = fig_r.add_subplot(1, 1, 1)
#     ax.plot(dict['time'][0],np.array(dict['r'][0]),label='Knudsen')
#     ax.plot(dict1['time'],np.array(dict1['rs']),label='$d^{2}$: $T_{s}$=Const')
#     ax.plot(dict2['time'],np.array(dict2['rs']),label='$d^{2}$: Infinite Conductivity')

#     locs,labels = yticks()
#     # yticks(locs, map(lambda x: "%.1f" % x, locs*1e9))
#     # text(0.0, 1.01, '1e-6', fontsize=10, transform = gca().transAxes)

#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel('$r$ [m]')
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
#     ax.legend()
#     if save:
#         fig_r.savefig('../Reports/Final_report/Figures/r_hom_vs_D2_1dT.pdf')

#     # fig_r_lin = plt.figure(figsize=(12,5))
#     # ax = fig_r_lin.add_subplot(1, 1, 1)
#     # ax.plot(dict['time'][0],dict['r'][0],label='Knudsen')
#     # ax.plot(dict1['time'],dict1['rs'],label='$d^{2}$: $T_{s}$=Const')
#     # ax.plot(dict2['time'],dict2['rs'],label='$d^{2}$: Infinite Conductivity')
#     # ax.set_xlabel('Time [s]')
#     # ax.set_ylabel('r [m]')
#     # # ax.set_xscale('log')
#     # # ax.set_yscale('log')
#     # ax.legend()
#     # if save:
#     #     fig_r_lin.savefig('../Reports/Final_report/Figures/r_hom_vs_D2_lin_1dT.pdf')

    
#     ax = fig_r_sq_lin.add_subplot(2, 1, i)
#     ax.set_title('d$T$={}K'.format(dT[i-1]),fontsize=17)
#     ax.plot(dict['time'][0],np.array(dict['r_sq'][0][0]),label='Knudsen')
#     ax.plot(dict1['time'],dict1['r_sq'],label='$d^{2}$: $T_{s}$=Const')
#     ax.plot(dict2['time'],dict2['r_sq'],label='$d^{2}$: Infinite Conductivity')
#     ax.yaxis.set_major_formatter(y_formatter)
#     if i==2:
#         ax.set_xlabel('Time [s]')
#     else:
#         ax.legend()
#     ax.set_ylabel('$r^{2}$ [m$^{2}$]')
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
   
#     if save:
#         fig_r_sq_lin.savefig('../Reports/Final_report/Figures/r_sq_hom_vs_D2_lin.pdf')
#     # fig_r_sq_lin.savefig('Plots/r_sq_hom_vs_D2_lin.png',transparent=True)
    
#     ax = fig_k_sq.add_subplot(2, 1, i)
#     ax.set_title('d$T$={}K'.format(dT[i-1]),fontsize=17)
#     ax.plot(dict['time'][0],dict['k_sq'][0],label='Knudsen')
#     ax.plot(dict1['time'],dict1['k_sq'][0],label='$d^{2}$: $T_{s}$=Const')
#     ax.plot(dict2['time'],dict2['k_sq'][0],label='$d^{2}$: Infinite Conductivity')
#     # ax.yaxis.set_major_formatter(y_formatter)
#     if i==2:
#         ax.set_xlabel('Time [s]')
#     else:
#         ax.legend(loc='lower right')
#     ax.set_ylabel('$K$ [m$^{2}$/s]')
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
    
#     if save:
#         fig_k_sq.savefig('../Reports/Final_report/Figures/k_sq_hom_vs_D2.pdf')

#     # fig_r_sq = plt.figure(figsize=(12,5))
#     # ax = fig_r_sq.add_subplot(1, 1, 1)
#     # ax.plot(dict['time'][0],dict['r_sq'][0][0],label='Knudsen')
#     # ax.plot(dict1['time'],dict1['r_sq'],label='$d^{2}$: $T_{s}$=Const')
#     # ax.plot(dict2['time'],dict2['r_sq'],label='$d^{2}$: Infinite Conductivity')
#     # ax.yaxis.set_major_formatter(y_formatter)
#     # ax.set_xlabel('Time [s]')
#     # ax.set_ylabel('$r^{2}$ [m$^{2}$]')
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
#     # ax.legend()
#     # if save:
#     #     fig_r_sq.savefig('../Reports/Final_report/Figures/r_sq_hom_vs_D2_1dT.pdf')
# plt.show()
# # endregion
# # Infinite Conduction zoom
# #region
# # save=True
# sns.set_style("whitegrid")
# plt.rc('font', size=17) 
# dict2 = main.D2(Q_ext=0,T_s=350+10,T_inf=350,delta_t=1e-10,r_s0=1e-6,endtime=1e-7,cond=True)
# dict=hom.Knudsen(T_s=373+10,T_inf=373,step=1e-10,endtime=1e-7,r0=1e-6)
# dict1 = main.D2(Q_ext=0,T_s=350+10,T_inf=350,delta_t=1e-10,r_s0=1e-6,endtime=1e-7,cond=False)
# fig_r_sq_lin = plt.figure(figsize=(13,6)) 
# ax = fig_r_sq_lin.add_subplot(1, 1, 1)
# y_formatter = ScalarFormatter(useOffset=False)
# ax.set_title('d$T$={}K'.format(10))
# ax.plot(dict['time'][0],np.array(dict['r_sq'][0][0])*1e6,label='Knudsen')
# ax.plot(dict1['time'],np.array(dict1['r_sq'])*1e6,label='$d^{2}$: $T_{s}$=Const')
# ax.plot(dict2['time'],np.array(dict2['r_sq'])*1e6,label='$d^{2}$: Infinite Conductivity',color='C2')
# ax.yaxis.set_major_formatter(y_formatter)

# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$r^{2}$ [$\mu$m$^{2}$]')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.legend()
# if save:
#     fig_r_sq_lin.savefig('../Reports/Final_report/Figures/r_sq_transient_initial.pdf')
# plt.show()
#endregion



#Knudsen vs Dropwise
#region


save=False
step=0.1
endtime=10
dT=[1,10]
# fig_r = plt.figure(figsize=(12,10))
# for i in range(1,3):
#     dict1=dropwise.Condense(T_wall=373-dT[i-1],T_inf=373,theta=175,endtime=endtime,step=step)
#     # print(dict1['r_min'])

#     # for i in Q:

#     dict=hom.Knudsen(T_s=373+dT[i-1],T_inf=373,step=step,endtime=endtime,r0=dict1['r_min'])

#     sns.set_style("whitegrid")
#     plt.rc('font', size=17) 
#     y_formatter = ScalarFormatter(useOffset=False)
    
    
#     ax = fig_r.add_subplot(2, 1, i)
#     ax.yaxis.set_major_formatter(y_formatter)
#     ax.set_title('d$T$={}K'.format(dT[i-1]),fontsize=17)
#     ax.plot(dict['time'][0],np.array(dict['r'][0])*1e6,label='Knudsen',color='C2')
#     ax.plot(dict1['time'][0],np.array(dict1['r'][0])*1e6,label='Dropwise',color='C3')
#     if i-1==1:
#         ax.set_xlabel('Time [s]')
#     else:
#         ax.legend()
#     ax.set_ylabel('r [$\mu$m]')
#     # ax.set_xscale('log')
#     # ax.set_yscale('log')
    
# # if save:
# fig_r.savefig('../Reports/Final_report/Figures/r_knud_vs_dropwise.pdf')
# plt.show()
# fig_r_lin = plt.figure(figsize=(12,5))
# ax = fig_r_lin.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['k_sq'][0],label='Knudsen',color='C2')
# ax.plot(dict1['time'][0],dict1['k_sq'],label='Dropwise',color='C3')
# # ax.scatter([2,4,6,8,10],[0.0039198773,0.0031113127,0.0027180143,0.0024694836,0.0022924678])
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$K$ [m$^{2}$/s$^{2}$]')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.legend()
# if save:
#     fig_r_lin.savefig('../Reports/Final_report/Figures/K_hom_vs_D2_lin.pdf')

# fig_r_sq_lin = plt.figure(figsize=(12,5))
# ax = fig_r_sq_lin.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['r_sq'][0][0],label='Knudsen',color='C2')
# ax.plot(dict1['time'][0],dict1['r_sq'][0][0],label='Dropwise',color='C3')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$r^{2}$ [m$^{2}$]')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.legend()
# if save:
#     fig_r_sq_lin.savefig('../Reports/Final_report/Figures/r_sq_hom_vs_D2_lin.pdf')



# fig_drdt = plt.figure(figsize=(12,5))
# ax = fig_drdt.add_subplot(1, 1, 1)
# ax.plot(dict['time'][0],dict['k_slope'][0],label='Knudsen',color='C2')
# ax.plot(dict1['time'][0],dict1['k_slope'][0],label='Dropwise',color='C3')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$\mathrm{d}r/\mathrm{d}t$ [m/s]')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.legend()
# if save:
#     fig_drdt.savefig('../Reports/Final_report/Figures/drdt_hom_vs_D2.pdf')
# plt.show()

save=False
sns.set_style("whitegrid")
# plt.rc('font', size=17) 
# fig_theta = plt.figure(figsize=(12,6))
# ax = fig_theta.add_subplot(1, 1, 1)
# for theta in range(30,120,20):
#     print(theta)
#     dict1=dropwise.Condense(T_wall=373-1,T_inf=373,theta=theta,endtime=100,step=0.0001,accom=0.02,lat_heat=2256400)
      
#     ax.plot(dict1['time'][0],np.array(dict1['r'][0])*1e6,color='C3')
# ax.annotate(r'Increasing $\theta$', xy=(10e-1, dict1['time'][0][-1]), xycoords='data',
#             xytext=(10e-2, dict1['time'][0][-1]-50), textcoords='offset points',
#             arrowprops=dict(arrowstyle='->', color='k'), ha='right', va='bottom',color='k')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$r$ [$\mu$m]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_xlim([1e-5,1e3])
# # ax.set_ylim([1e-8,1e10])
# # ax.legend()
# plt.show()
# if save:
#     fig_theta.savefig('../Reports/Final_report/Figures/dropwise_theta.pdf')


plt.rc('font', size=17) 
fig_theta = plt.figure(figsize=(10,10))
ax = fig_theta.add_subplot(2, 1, 1)
i=-1
for a in [0.02,0.04,1]:
    i+=1
    dict1=dropwise.Condense(T_wall=373-10,T_inf=373,theta=85,endtime=1,step=0.0001,accom=a,lat_heat=2256400)

    color='C{}'.format(i) 
    label='$\hat{fname}$={value}'.format(fname='\sigma',value=a)
    print(label)
    ax.plot(dict1['time'][0],np.array(dict1['r'][0])*1e6,color=color,label=label)

ax.set_xlabel('Time [s]')
ax.set_ylabel('$r$ [$\mu$m]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim([1e-4,1e1])
# ax.set_ylim([1e-8,1e10])
ax.legend()
# plt.show()
# if save:
#     fig_theta.savefig('../Reports/Final_report/Figures/dropwise_h_i.pdf')

# plt.rc('font', size=17) 
# fig_theta2 = plt.figure(figsize=(12,6))
ax = fig_theta.add_subplot(2, 1, 2)
i=-1
for a in [0.02,0.04,1]:
    i+=1
    dict1=dropwise.Condense(T_wall=373-10,T_inf=373,theta=85,endtime=1,step=0.0001,accom=a,lat_heat=2256400)

    color='C{}'.format(i) 
    label='$\hat{fname}$={value}'.format(fname='\sigma',value=a)
    print(label)
    ax.plot(np.array(dict1['r'][0])*1e6,np.array(dict1['k_slope'][0])*1e6,color=color,label=label)

ax.set_xlabel(r'$r$ [$\mu$m]')
ax.set_ylabel(r'$\mathrm{d}r/\mathrm{d}t$ [$\mu$m/s]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim([1e-4,1e1])
# ax.set_ylim([1e-8,1e10])
# ax.legend()
plt.show()
if save:
    fig_theta.savefig('../Reports/Final_report/Figures/dropwise_sigma_drdt.pdf')
fig_theta.savefig('Plots/dropwise_sigma_drdt.png',transparent=True)
#endregion