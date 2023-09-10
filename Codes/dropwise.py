
def Condense(T_inf= 350,T_wall=340,total_pressure=101325,delta_t=10**(-1),theta=86,endtime=1,step=1e-6,accom=0.04,lat_heat=2429.8*1e3):
    import functions as func
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.integrate import RK45
    #init var_dict
    var_dict=func.var_dict(time=[],delta_t=delta_t,theta=theta,r_e=[],n_r=[],r_max=[],r=[],r_mod=[],k_slope=[],k_sq=[],r_sq=[],q=[])
    #Evaluation of the surface tension [Jasper, J. J., J.Phys. Chem. Ref. Data, 1 (1972): 841] 
    var_dict['sigma']=0.06 #N/m
    var_dict['r_min']=(2*var_dict['sigma']*T_inf/func.get_rho_l(T_inf))/(lat_heat*(T_inf-T_wall))
    
    r0=var_dict['r_min']*1.01
    # r0=np.sqrt(0.037/1e9)
    # r0=np.sqrt(0.037/1e10)
    # print('min droplet radius: {}'.format(var_dict['r_min']))
    # print('Surface tension: {}'.format(var_dict['sigma']))
    #thermal conductivity coeff copper
    var_dict['k_copp']=0.677 #W/mK
    v_vl=1.673-0.001042
    #interface convection coeff
    var_dict['h_int']=(2*accom/(2-accom))*(lat_heat**(2))/(T_inf*v_vl)*np.sqrt(0.018/(2*np.pi*8.3144*T_inf))
    var_dict['r_max']=np.sqrt(((6*np.sin(theta*np.pi/180)**(2))/((2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3))))*(var_dict['sigma'])/(958*9.81))
    # print('h_int: {}'.format(var_dict['h_int']))
    F = lambda t, r: (4*(T_inf-T_wall)/(958.65*lat_heat))*((1-(var_dict['r_min']/r))/((2/var_dict['h_int'])+(r*(1-np.cos(theta*np.pi/180))/var_dict['k_copp'])))*((1-np.cos(theta*np.pi/180))/(2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3)))
    # F = lambda t, r: (((T_inf-T_wall)/(960*2256.4*10**(3)*(2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3))))*(1-(var_dict['r_min']/r)))/(((theta*np.pi/180)/(4*var_dict['k_copp']*np.sin(theta*np.pi/180)**(2)))*r+(1/(2*var_dict['h_int']*(1-np.cos(theta*np.pi/180)))))
    t_eval = np.arange(0, endtime, step)
    # t_eval=None
    solution = solve_ivp(fun=F, t_span=(0,endtime) , y0=[r0],method='RK45',t_eval=t_eval)
    # solution = RK45(fun=F, t_bound=endtime , y0=[r0],t0=0)
    # plt.plot(solution.t,solution.y[0])
    # plt.show()
    var_dict['r'].append(solution.y[0])
    var_dict['r_sq'].append(np.power(var_dict['r'],2))
    var_dict['time'].append(solution.t)
    var_dict['k_slope'].append(np.diff(np.array(solution.y[0]),append=0)/(solution.t[1]-solution.t[0]))
    var_dict['k_slope'][0][-1]=var_dict['k_slope'][0][-2]
    # var_dict['k_sq'].append(np.diff(np.array(solution.y[0])**(2),append=0)/(solution.t[1]-solution.t[0]))
    # var_dict['k_sq'][0][-1]=var_dict['k_sq'][0][-2]

    r_array=np.array(solution.y[0])
    r_e=1/np.sqrt(4*(0.037/var_dict['r_min']**(2)))
    var_dict['r_e']=r_e
    var_dict['n_r']=(1/(3*np.pi*var_dict['r_max']*r_array**(2)))*(r_array/var_dict['r_max'])**(-2/3)
    # first_part=N_e*((4*(T_inf-T_wall)/(958.65*lat_heat))*((1-(var_dict['r_min']/r_e))/((2/var_dict['h_int'])+(r_e*(1-np.cos(theta*np.pi/180))/var_dict['k_copp'])))*((1-np.cos(theta*np.pi/180))/(2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3))))/((4*(T_inf-T_wall)/(958.65*lat_heat))*((1-(var_dict['r_min']/r_array))/((2/var_dict['h_int'])+(r_array*(1-np.cos(theta*np.pi/180))/var_dict['k_copp'])))*((1-np.cos(theta*np.pi/180))/(2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3))))

    # var_dict['q']=(T_inf-T_wall)*(1-var_dict['r_min']/r_array)*(1/(2*np.pi*np.power(r_array,2)*var_dict['h_int']*(1-np.cos(theta)))+(r_array*(1-np.cos(theta)))/(4*np.pi*var_dict['k_copp']*np.power(r_array,2)*(1-np.cos(theta))))**(-1)
    var_dict['q']=lat_heat*func.get_rho_l(T_inf)*np.pi*r_array**(2)*(2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3))*(4*(T_inf-T_wall)/(958.65*lat_heat))*((1-(var_dict['r_min']/r_array))/((2/var_dict['h_int'])+(r_array*(1-np.cos(theta*np.pi/180))/var_dict['k_copp'])))*((1-np.cos(theta*np.pi/180))/(2-3*np.cos(theta*np.pi/180)+np.cos(theta*np.pi/180)**(3)))
    # print('new radius: {}'.format(var_dict['r']))

    return var_dict

