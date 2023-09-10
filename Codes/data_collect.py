import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import main 
# Q = np.linspace(0.000001,0.0003,10)
pressures=np.linspace(101300,401300,50)
Temp_diff=np.linspace(1,40,50)
# pressures=[101300]
# Temp_diff=[20]
df_main = pd.DataFrame()
# for i in Q:
for p in pressures:
    for dT in Temp_diff:
        dict1 = main.D2(Q_ext=0,total_pressure=p,T_s=372-dT)
        delta_t=dict1['delta_t']
        # k=np.diff(np.array(dict1['rs'])**(2),append=0)/delta_t
        r=dict1['rs']
        time=np.array(dict1['time'])
        time_to_r=time[np.abs(r-r[-1])<0.000000000001][0]
        dummy_dict={'Ambient Pressure [Pa]':[p],
                    'dT [K]':[dT],
                    '$r$ [m]':[r[-1]],
                    'Time_r [s]':[time_to_r]} 
                                
                    
        
        df = pd.DataFrame(dummy_dict)
        df_main = pd.concat([df_main, df])

df_main.to_csv('Data/r.csv')