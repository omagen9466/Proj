
def Knudsen(T_inf= 373,T_s=363,total_pressure=101325,endtime=1,step=1e-6,r0=1e-9,lat_heat=2264705):
    import functions as func
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    #init var_dict
    var_dict=func.var_dict(time=[],r=[],r_mod=[],k_slope=[],k_sq=[],r_sq=[],q=[])
    total_pressure=func.get_sat_press(T_inf,1)*np.exp((2*0.06)/(950*461*T_s*r0))
      
    
 
    
    var_dict['k_g']=0.02457 #W/mK
    #interface convection coeff
    #for knudsen
    l=1.5*0.2938e-3*np.sqrt(T_inf*461)/total_pressure


    F = lambda t, r: (var_dict['k_g']/(r*(1+3.18*(l/2*r))))*((T_s-T_inf)/(lat_heat*958.35))
    t_eval = np.arange(0, endtime, step)
    solution = solve_ivp(fun=F, t_span=[0,endtime] , y0=[r0],method='RK45',t_eval=t_eval)
    # plt.plot(solution.t,solution.y[0])
    # plt.show()
    var_dict['r'].append(solution.y[0])
    var_dict['r_sq'].append(np.power(var_dict['r'],2))
    var_dict['time'].append(solution.t)
    var_dict['k_slope'].append(np.diff(np.array(solution.y[0]),append=0)/(solution.t[1]-solution.t[0]))
    var_dict['k_slope'][0][-1]=var_dict['k_slope'][0][-2]
    var_dict['k_sq'].append(np.diff(np.array(solution.y[0])**(2),append=0)/(solution.t[1]-solution.t[0]))
    var_dict['k_sq'][0][-1]=var_dict['k_sq'][0][-2]

    r_array=np.array(solution.y[0])
   
    var_dict['q']=np.pi*r_array**(2)*2264705*func.get_rho_l(T_inf)*(var_dict['k_g']/(r_array*(1+3.18*(l/2*r_array))))*((T_inf-T_s)/(2264705*958.35))
    # print('new radius: {}'.format(var_dict['r']))

    return var_dict

