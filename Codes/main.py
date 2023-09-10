"""main code: initializes the calculation of the evap process"""
def D2(T_s=320,T_inf= 373,total_pressure=101326,r_s0 = 10**(-6),U=0.00000001,delta_t=10**(-9),Q_ext=11**(-3),endtime=1e-9,cond=True):

    import functions as func
    import numpy as np
    import matplotlib.pyplot as plt
    #init var_dict
    var_dict=func.var_dict(Ps=[func.get_sat_press(T_s,total_pressure)],rs=[r_s0],m=[np.pi*func.get_rho_l(T_s)*r_s0**(3)*4/3],Y_s=[],
                           Bt=[],T_s=[T_s],time=[0],delta_t=delta_t,rho_l=[func.get_rho_l(T_s)],k_slope=[],k_sq=[],r_sq=[r_s0**(2)],q_drop=[],m_dot=[])
    """The steps taken follow (Abramzon 1989) calculation"""
    #1: calculate mass fractions at droplet surface and ambient

  
    # var_dict['Y_inf']=func.get_Y(func.get_sat_press(T_inf,total_pressure),total_pressure)
    var_dict['Y_inf']=func.get_super_Y(T_s,total_pressure,r_s0)
   
    # print(var_dict['Y_inf'])
    var_dict['Y_s'].append(func.get_Y(var_dict['Ps'][-1],total_pressure))
    print(var_dict['Y_s'])
    Y_s=var_dict['Y_s'][-1]
    m=var_dict['m'][-1]
    for i in range(1,int(endtime/delta_t)):
        
        # print('Y_inf: {}'.format(var_dict['Y_inf']))
 
        # print('Y_s: {}'.format(var_dict['Y_s'][-1]))
        #2 calculate one third law for required parameters
        var_dict['rho_g']=func.get_one_third(func.get_rho_g(T_inf,total_pressure),func.get_rho_g(T_s,total_pressure)) 
        # print('rho_g: {}'.format(var_dict['rho_g']))
        # var_dict['Cp_w']=func.get_one_third(func.get_Cp(T_inf,'w',total_pressure),func.get_Cp(T_s,'w',total_pressure))
        var_dict['Cp_w']=4215
        # print('Cp_w: {}'.format(var_dict['Cp_w']))
        var_dict['Cp_g']=func.get_one_third(func.get_Cp(T_inf,0,total_pressure),func.get_Cp(T_s,0,total_pressure))
        # print('Cp_g: {}'.format(var_dict['Cp_g']))
        var_dict['k_g']=func.get_one_third(func.get_k(T_inf,total_pressure),func.get_k(T_s,total_pressure))
        # print('k_g: {}'.format(var_dict['k_g']))
        var_dict['mu_g']=func.get_one_third(func.get_mu(T_inf,total_pressure),func.get_mu(T_s,total_pressure))
        # print('mu_g: {}'.format(var_dict['mu_g']))
        var_dict['D']=func.get_one_third(func.get_mass_diffusivity(T_inf),func.get_mass_diffusivity(T_s))
        # print('D: {}'.format(var_dict['D']))
        var_dict['Le']=var_dict['k_g']/(var_dict['D']*var_dict['rho_g']*var_dict['Cp_g'])
        # print('Le: {}'.format(var_dict['Le']))
        var_dict['Pr']=(var_dict['Cp_g']*var_dict['mu_g'])/var_dict['k_g']
        # print('Pr: {}'.format(var_dict['Pr']))
        var_dict['Sc']=var_dict['mu_g']/(var_dict['rho_g']*var_dict['D'])
        # print('Sc: {}'.format(var_dict['Sc']))
        # var_dict['lambda']=var_dict['k_g']/(var_dict['Cp_g']*var_dict['Cp_g'])
    
        #3 calculate one third law for required parameters
        var_dict['Re']=2*func.get_rho_g(T_inf,total_pressure)*U*r_s0/var_dict['mu_g']
        # print(var_dict['Re'])
        # Nu0=2+0.552*(var_dict['Re']**(0.5))*var_dict['Pr']**(1/3)
        # Sh0=2+0.552*(var_dict['Re']**(0.5))*var_dict['Sc']**(1/3)
        Nu0=1+(1+var_dict['Re']*var_dict['Pr'])**(1/3)
        Sh0=1+(1+var_dict['Re']*var_dict['Sc'])**(1/3)
        #4 calculate spalding mass transfer, radius, mass condensation rate
        
        var_dict['Bm']=func.get_spalding_mass(Y_s,var_dict['Y_inf']) 
        F_Bm=(1+var_dict['Bm'])**(0.7)*(np.log(1+var_dict['Bm'])/var_dict['Bm'])
        Sh_star=2+(Sh0-2)/F_Bm
        # print('Bm: {}'.format(var_dict['Bm']))
        m_dot=-4*np.pi*var_dict['rho_g']*var_dict['D']*r_s0*Sh_star*np.log(1+var_dict['Bm'])
        # print('The constant mass flow rate is: {} mgrams per sec'.format(var_dict['m_dot']*10**6))

        #5 calculate Bt and Ft
        if var_dict['time'][-1]==0:
            var_dict['Bt'].append(func.get_spalding_temp(var_dict['Cp_w'],T_inf,T_s,func.get_latent_h(T_s),m_dot)) 

        # print(' Bt before loop: {}'.format(var_dict['Bt']))
        for j in range(10):
            F_Bt =(1+var_dict['Bt'][-1])**(0.7)*(np.log(1+var_dict['Bt'][-1])/var_dict['Bt'][-1])  
            Nu_star=2+(Nu0-2)/F_Bt
            phi=(var_dict['Cp_w']/var_dict['Cp_g'])*(Sh_star/Nu_star)*(1/var_dict['Le'])
            # print('!!!!!!{}!!!!!!'.format(phi))
            Bt=(1+var_dict['Bm'])**(phi)-1
            # print('!!!!!!{}!!!!!!'.format(Bt))
            if np.abs(Bt-var_dict['Bt'][-1])<0.0001:
                
                break
            var_dict['Bt'][-1]=Bt
        # print('Iterations for Bt: {}'.format(j+1))  
        
        # print(' Bt after loop: {}'.format(Bt))
        #7 Calculate the heat leaving the droplet
        q_drop=-m_dot*((var_dict['Cp_w']*(T_inf-T_s)/var_dict['Bt'][-1])-func.get_latent_h(T_s))
        # print('q drop: {}'.format(q_drop))
        # print('lat heat: {}'.format(var_dict['m_dot']))
        # print(func.get_rho_l(var_dict['T_s'][-1]))
        r_s0=((3/4)/np.pi/func.get_rho_l(T_s)*(m+m_dot*delta_t))**(1/3)
        m=np.pi*func.get_rho_l(T_s)*r_s0**(3)*4/3
        # print(var_dict['rs'])
        
        # if type(Q_ext) is list:# or np.ndarray: 
        #     T_s=(q_drop-Q_ext[i])*6*delta_t/((m+var_dict['m_dot']*delta_t)*var_dict['Cp_w'])+T_s
        if cond==True:
            T_s=(q_drop-Q_ext)*6*delta_t/((m+m_dot*delta_t)*var_dict['Cp_w'])+T_s
        rho_l=func.get_rho_l(T_s)
        # print('rho_l: {}'.format(rho_l))
        
        var_dict['Ps']=func.get_sat_press(T_s,total_pressure)
        
        Y_s=func.get_Y(var_dict['Ps'],total_pressure)
        #save every 1000 iterations to save memory
        if i%1==0:
            var_dict['rs'].append(r_s0)
            var_dict['r_sq'].append(r_s0**(2))
            var_dict['m'].append(m)
            var_dict['T_s'].append(T_s)
            var_dict['time'].append(i*delta_t)
            var_dict['rho_l'].append(rho_l)
            var_dict['Y_s'].append(Y_s)
            var_dict['q_drop'].append(q_drop)
            var_dict['m_dot'].append(m_dot)
    var_dict['k_slope'].append(np.diff(np.array(var_dict['rs']),append=0)/(var_dict['time'][1]-var_dict['time'][0]))
    var_dict['k_slope'][0][-1]=var_dict['k_slope'][0][-2]
    var_dict['k_sq'].append(np.diff(np.array(var_dict['rs'])**(2),append=0)/(var_dict['time'][1]-var_dict['time'][0]))
    var_dict['k_sq'][0][-1]=var_dict['k_sq'][0][-2]
    
    return var_dict

 
    






