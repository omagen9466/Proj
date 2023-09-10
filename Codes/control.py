import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import main 
import dropwise
import hom
#Following region only D2 plots
#region
# Q = [1e-8]
# pressures=[101300]
# Temp_diff=[2]

# df_main = pd.DataFrame()
# for i in Q:
#     for p in pressures:
#         for dT in Temp_diff:
#             dict1 = main.D2(Q_ext=i,total_pressure=p,T_s=372-dT,delta_t=10**(-15),r_s0=10**(-9))
#             delta_t=dict1['delta_t']
#             k=np.diff(np.array(dict1['rs'])**(2),append=0)/delta_t
#             dummy_dict={'Time [s]':pd.Series([dict1['time']]),
#                         '$T_{s}$ [K]':pd.Series([dict1['T_s']]),
#                         '$r_{s}$ [m]':pd.Series([dict1['rs']]),
#                         '$Y_{s}$':pd.Series([dict1['Y_s']]),
#                         '$k$ [m/s]':pd.Series([k]),
#                         '$q_{out}$ [W]': [i],
#                         'Ambient Pressure [Pa]':[p],
#                         'dT [K]':[dT],
#                         'rho_l [kg/m3]':pd.Series([dict1['rho_l']])}
            
#             df = pd.DataFrame(dummy_dict)
#             df_main = pd.concat([df_main, df])

# df_main = df_main.set_index(['$q_{out}$ [W]','Ambient Pressure [Pa]','dT [K]']).apply(pd.Series.explode).reset_index()

# sns.set_style("whitegrid")
# g=sns.relplot(x='Time [s]'
#               ,y="$T_{s}$ [K]",
#               data=df_main,
#               kind="line",
#               col= 'dT [K]',
#               row='Ambient Pressure [Pa]',
#               hue='$q_{out}$ [W]',
#               palette='tab10',
#               facet_kws=dict(sharey=False,sharex=False,despine=False))
# sns.move_legend(g, "lower right", bbox_to_anchor=(0.88, 0.55),frameon=True)
# for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
#     ax.text(0.02, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/T_s.pdf')
# plt.show()

# g=sns.relplot(x='Time [s]'
#               ,y="$r_{s}$ [m]",
#               data=df_main,
#               kind="line",
#               col= 'dT [K]',
#               row='Ambient Pressure [Pa]',
#               hue='$q_{out}$ [W]',
#               palette='tab10',
#               facet_kws=dict(sharey=False,sharex=False,despine=False))
# sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
#     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')
# plt.show()

# g=sns.relplot(x='Time [s]',
#               y="$k$ [m/s]",
#               data=df_main,
#               kind="line",
#               col= 'dT [K]',
#               row='Ambient Pressure [Pa]',
#               hue='$q_{out}$ [W]',
#               palette='tab10',
#               facet_kws=dict(sharey=False,sharex=False,despine=False))
# sns.move_legend(g, "lower right", bbox_to_anchor=(0.85, 0.82),frameon=True)
# g.set(xlim=(-1e-7, 7.5e-7),ylim=(-1.5e-7,1.5e-7))
# for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
#     ax.text(-0.07, 1.09, label, transform=ax.transAxes, fontsize=11, va='top')
# g.savefig('../Reports/Progress_report/Figures/k.pdf')
# plt.show()

# g=sns.relplot(x='Time [s]'
#               ,y="rho_l [kg/m3]",
#               data=df_main,
#               kind="line",
#               col= 'dT [K]',
#               row='Ambient Pressure [Pa]',
#               hue='$q_{out}$ [W]',
#               palette='tab10',
#               facet_kws=dict(sharey=False,sharex=False,despine=False))
# sns.move_legend(g, "lower right", bbox_to_anchor=(0.88, 0.55),frameon=True)
# for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
#     ax.text(0.02, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# g.savefig('../Reports/Progress_report/Figures/T_s.pdf')
# plt.show()
#endregion

#Following region D2 Dropwise compare
#region

# Q = [0]
# # Q=[2e-4]
# Temp_diff=[1]
# df_main = pd.DataFrame()
# df_condense= pd.DataFrame()
# # for i in Q:
# dropwise_step=1e-10 
# for dT in Temp_diff:   
#   for QQ in Q:
#       dict=dropwise.Condense(T_wall=373-dT,theta=30,endtime=5e-7,step=dropwise_step)
#       # dict1 = main.D2(Q_ext=dict['q'],T_s=372-dT,delta_t=10**(-15),r_s0=dict['r_min']*1.001)
#       # print('Heat out of droplet: {}'.format(dict['q'][-1]))
      
#       # dummy_dict_c={'Time [s]':pd.Series([dict['time'][0]]),
#       #             # '$r_{s}$ [m]':pd.Series([dict['r'][0]]),
#       #             # '$r_{s}^2$ [m]':pd.Series([dict['r_sq'][0][0]]),
#       #             '$k$ [m/s]':pd.Series([dict['k_slope'][0]]),
#       #             # '$k^2$ [m/s]':pd.Series([dict['k_sq'][0]]),
#       #             'dT [K]':[Temp_diff[0]],
#       #             'Q [W]':[QQ],
#       #             'Model':['Dropwise']}
      
#       dummy_dict={'Time [s]':pd.Series([dict['time'][0]]),
#                   # '$r_{s}$ [m]':pd.Series([dict1['rs']]),
#                   # '$r_{s}^2$ [m]':pd.Series([dict1['r_sq']]),
#                   '$k$ [m/s]':pd.Series([dict['k_slope'][0]]),
#                   # '$k^2$ [m/s]':pd.Series([dict1['k_sq'][0]]),
#                   'Q [W]':[QQ],
#                   'dT [K]':[dT],
#                   'Model':['D2']}
      
#       # df_c = pd.DataFrame(dummy_dict_c)
#       # df_condense = pd.concat([df_condense, df_c])
#       df = pd.DataFrame(dummy_dict)
#       df_main = pd.concat([df_main, df])
# # df_main = pd.concat([df_main, df_c])
# print('Finished creating DF')
# df_main = df_main.set_index(['dT [K]','Model','Q [W]']).apply(pd.Series.explode).reset_index()

# sns.set_style("whitegrid")

# # TT=sns.relplot(x='Time [s]',
# #               y="$r_{s}$ [m]",
# #               data=df_main,
# #               hue='Model',
# #               kind="line",
# #               palette='tab10',
# #               style='Q [W]'
# #               )
# # # sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# # # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# # #     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')
# # TT.set(xscale="log")
# # TT.set(yscale="log")

# # plt.show()

# # TS=sns.relplot(x='Time [s]',
# #               y="$r_{s}^2$ [m]",
# #               data=df_main,
# #               hue='Model',
# #               kind="line",
# #               palette='tab10',
# #               style='Q [W]'
# #               )
# # # sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# # # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# # #     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')
# # TS.set(xscale="log")
# # TS.set(yscale="log")

# # plt.show()

# g=sns.relplot(x='Time [s]',
#               y="$k$ [m/s]",
#               data=df_main,
#               hue='Model',
#               kind="line",
#               palette='tab10',
#               style='Q [W]'
#               )#,
#             #   facet_kws=dict(sharey=False,sharex=False,despine=False))
# # sns.move_legend(g, "lower right", bbox_to_anchor=(0.85, 0.82),frameon=True)
# # g.set(xlim=(-1e-7, 7.5e-7),ylim=(-1.5e-7,1.5e-7))
# # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# #     ax.text(-0.07, 1.09, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/k.pdf')
# g.set(xscale="log")
# g.set(yscale="log")
# # g.savefig('k_every_it.png')
# plt.show()

# f=sns.relplot(x='Time [s]',
#               y="$k^2$ [m/s]",
#               data=df_main,
#               hue='Model',
#               kind="line",
#               palette='tab10',
#               style='Q [W]'
#               )#,
#             #   facet_kws=dict(sharey=False,sharex=False,despine=False))
# # sns.move_legend(g, "lower right", bbox_to_anchor=(0.85, 0.82),frameon=True)
# # g.set(xlim=(-1e-7, 7.5e-7),ylim=(-1.5e-7,1.5e-7))
# # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# #     ax.text(-0.07, 1.09, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/k.pdf')
# f.set(xscale="log")
# f.set(yscale="log")
# plt.show()
# plt.figure()
# plt.plot(dict1['time'][1:],dict1['q_drop'])
# # plt.plot(dict['time'][0],dict['q'])


# plt.show()

#endregion

#Following region only Dropwise plots
#region
# Temp_diff=[1]
# Thetas=[90,95,100,105,110,120,130,140,150,160,170]
# df_main = pd.DataFrame()
# df_condense= pd.DataFrame()
# # for i in Q:
# for Th in Thetas:    
#     for dT in Temp_diff:
        
#         dict=dropwise.Condense(T_wall=372-dT,theta=Th)
        
#         dummy_dict_c={'Time [s]':pd.Series([dict['time'][0]]),
#                     '$r_{s}$ [m]':pd.Series([dict['r'][0]]),
#                     '$q$ [W]':pd.Series([dict['q']]),
#                     'dT [K]':[dT],
#                     'Model':['Dropwise'],
#                     'Theta':[Th]}
        

        
#         df_c = pd.DataFrame(dummy_dict_c)
#         df_condense = pd.concat([df_condense, df_c])


# df_condense = df_condense.set_index(['dT [K]','Model','Theta']).apply(pd.Series.explode).reset_index()

# sns.set_style("whitegrid")

# g=sns.relplot(x='Time [s]',
#               y='$q$ [W]',
#               data=df_condense,
#               hue='Theta',
#               kind="line",
#               palette='tab10',
#               style='dT [K]'
#               )
# # g.set(xscale="log")
# # g.set(yscale="log")
# # sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# #     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')

# plt.show()
#endregion

#Following region only Knudsen plots
#region
Temp_diff=[1]
Kn_list=[0.01]

df_knudsen= pd.DataFrame()
df_main= pd.DataFrame()
# for i in Q:
step=1e-13
endtime=5e-7
for dT in Temp_diff:
    for Kn1 in Kn_list:
      dict=hom.Knudsen(T_s=372-dT,Kn=Kn1,step=step,endtime=endtime,r0=1e-6)
      dict1 = main.D2(Q_ext=dict['q'],T_s=372-dT,delta_t=step,r_s0=1e-6,endtime=endtime)
      dummy_dict_k={'Time [s]':pd.Series([dict['time'][0]]),
                  '$r_{s}$ [m]':pd.Series([dict['r'][0]]),
                  '$r_{s}^2$ [m]':pd.Series([dict['r_sq'][0][0]]),
                  '$k$ [m/s]':pd.Series([dict['k_slope'][0]]),
                  '$k^2$ [m/s]':pd.Series([dict['k_sq'][0]]),
                  'dT [K]':[dT],
                  'Model':['Knudsen']
                  }
      dummy_dict={'Time [s]':pd.Series([dict1['time']]),
                  '$r_{s}$ [m]':pd.Series([dict1['rs']]),
                  '$r_{s}^2$ [m]':pd.Series([dict1['r_sq']]),
                  '$k$ [m/s]':pd.Series([dict1['k_slope'][0]]),
                  '$k^2$ [m/s]':pd.Series([dict1['k_sq'][0]]),
                  'dT [K]':[dT],
                  'Model':['D2']}
      

      
      df_k = pd.DataFrame(dummy_dict_k)
      df_knudsen = pd.concat([df_knudsen, df_k])
      df = pd.DataFrame(dummy_dict)
      df_main = pd.concat([df_main, df])
df_main = pd.concat([df_main, df_knudsen])
print('Finished creating DF')

df_main = df_main.set_index(['dT [K]','Model']).apply(pd.Series.explode).reset_index()
print(df_main['Time [s]'])
sns.set_style("whitegrid")

g=sns.relplot(x='Time [s]',
              y='$r_{s}$ [m]',
              data=df_main,
              kind="line",
              hue='Model',
              palette='tab10',
              style='dT [K]'
              )
# g.set(xscale="log")
# g.set(yscale="log")
# sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
#     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# g.savefig('../Reports/Progress_report/Figures/r_s.pdf')

plt.show()

# f=sns.relplot(x='Time [s]',
#               y='$r_{s}^2$ [m]',
#               data=df_main,
#               kind="line",
#               hue='Model',
#               palette='tab10',
#               style='dT [K]'
#               )
# # g.set(xscale="log")
# # g.set(yscale="log")
# # sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# #     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')

# plt.show()

# h=sns.relplot(x='Time [s]',
#               y='$k$ [m/s]',
#               data=df_main,
#               kind="line",
#               hue='Model',
#               palette='tab10',
#               style='dT [K]'
#               )
# # g.set(xscale="log")
# # g.set(yscale="log")
# # sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# #     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')

# plt.show()

# e=sns.relplot(x='Time [s]',
#               y='$k^2$ [m/s]',
#               data=df_main,
#               kind="line",
#               hue='Model',
#               palette='tab10',
#               style='dT [K]'
#               )
# # g.set(xscale="log")
# # g.set(yscale="log")
# # sns.move_legend(g, "lower right", bbox_to_anchor=(0.65, 0.82),frameon=True)
# # for ax, label in zip(g.axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
# #     ax.text(-0.07, 1.05, label, transform=ax.transAxes, fontsize=11, va='top')
# # g.savefig('../Reports/Progress_report/Figures/r_s.pdf')

# plt.show()
#endregion