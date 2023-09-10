###This class will includes function or the system of equations###
import numpy as np
from CoolProp.CoolProp import PropsSI
water_mol_weight=18.02*10**(-3) #kg/mol
air_mol_weight=28.9647*10**(-3) #kg/mol

def var_dict(**kwargs):
    return kwargs

# def get_sat_press(T,P):
#     lat_enth=40657 #j/mol
#     temp=99.974+273.15 #K
#     p=P #Pa
#     R=8.31446261815324 #J/mol*K
#     return p*np.exp((-lat_enth/R)*((1/T)-(1/temp)))
#Using IF97
def get_sat_press(T,P):
    p=PropsSI('P','T',T,'Q',0,'IF97::Water')
    return p
# def get_int_press(T_inf,T_s,r,P_atm):
#     P=get_sat_press(T_s,1)
#     x=P/P_atm
#     # rho_g=get_rho_g(T_s,P)
#     rho_l=PropsSI('D','T',T_s,'Q',0,'IF97::Water')
    
#     sigma=(75.83-(T_s-273)*0.1477)*10**(-3)
#     # temp=99.974+273.15 #K
   
#     R=8.31446261815324 #J/mol*K
#     return P*np.exp((2*sigma*water_mol_weight)/(R*T_s*rho_l*r))

# def get_int_press(T_inf,T_s,r):
#     P=get_sat_press(T_inf,1)
    
#     rho_l=PropsSI('D','T',T_s,'Q',0,'IF97::Water')
    
#     sigma=(75.83-(T_s-273)*0.1477)*10**(-3)
#     # temp=99.974+273.15 #K
   
#     R=8.31446261815324 #J/mol*K
#     return P*np.exp((2*sigma*water_mol_weight)/(R*T_s*rho_l*r))

def get_super_Y(T,p,R):
    sigma=(75.83-(T-273)*0.1477)*10**(-3)
    
    Y=get_Y(get_sat_press(T,p),p)*np.exp((2*water_mol_weight*sigma)/(6.022e23*1.3806e-23*T*get_rho_l(T)*R))
    return Y
    

def get_rho_g(T,P):
    x=get_sat_press(T,P)/P
    rho_v=x*P/(T*461.52)
    rho_a=(1-x)*P/(T*287.05)
    Y=get_Y(get_sat_press(T,P),P)
    return rho_v*Y+rho_a*(1-Y)

def get_rho_v(T,P):
    x=get_sat_press(T,P)/P
    rho_v=x*get_sat_press(T,P)/(T*461.52)
    return rho_v

def get_rho_l(T):
    return 998.21/(1+0.0002*(T-293))

def get_one_third(x_inf,x_s):
    return x_s+(1/3)*(x_inf-x_s)

def get_mass_diffusivity(T):
    A=1.859*(10**(-3))
    sigma=1.005*(0.5*(3.72+2.641))**2
    molma=np.sqrt((1/(water_mol_weight*10**3))+(1/(air_mol_weight*10**3)))
    # return 0.0000399
    return A*molma*(T**(1.5))/(sigma)*10**(-4)
    #https://www.vcalc.com/wiki/vCalc/Mass+Diffusivity+%28gas%29

# def get_Cp(T,s,P):
#     C1_w=0.33363 * 10**(5)
#     C2_w=0.2679 * 10**(5)
#     C3_w=2.6105* 10**(3)
#     C4_w=0.08896* 10**(5)
#     C5_w=1169
#     Cp_w=C1_w+C2_w*((C3_w/T)/np.sinh(C3_w/T))**(2)+C4_w*((C5_w/T)/np.cosh(C5_w/T))**(2)
#     Cp_w=Cp_w/(water_mol_weight*10**(3))
#     if s=='w':
#         return Cp_w
#     else:
#         C1_a=0.28958 * 10**(5)
#         C2_a=0.0939 * 10**(5)
#         C3_a=3.012* 10**(3)
#         C4_a=0.0758* 10**(5)
#         C5_a=1484
#         Cp_a = C1_a+C2_a*((C3_a/T)/np.sinh(C3_a/T))**(2)+C4_a*((C5_a/T)/np.cosh(C5_a/T))**(2)
#         Cp_a=Cp_a/(air_mol_weight*10**(3))
#         Y=get_Y(get_sat_press(T,P),P)
#         return Cp_w*Y+Cp_a*(1-Y)
# #perry's chemical book function page 224

def get_Cp(T,s,P):
    C1_w=0.33363 * 10**(5)
    C2_w=0.2679 * 10**(5)
    C3_w=2.6105* 10**(3)
    C4_w=0.08896* 10**(5)
    C5_w=1169
    Cp_w=C1_w+C2_w*((C3_w/T)/np.sinh(C3_w/T))**(2)+C4_w*((C5_w/T)/np.cosh(C5_w/T))**(2)
    Cp_w=Cp_w/(water_mol_weight*10**(3))
    if s=='w':
        return 4215
    else:
        C1_a=0.28958 * 10**(5)
        C2_a=0.0939 * 10**(5)
        C3_a=3.012* 10**(3)
        C4_a=0.0758* 10**(5)
        C5_a=1484
        Cp_a = C1_a+C2_a*((C3_a/T)/np.sinh(C3_a/T))**(2)+C4_a*((C5_a/T)/np.cosh(C5_a/T))**(2)
        Cp_a=Cp_a/(air_mol_weight*10**(3))
        Y=get_Y(get_sat_press(T,P),P)
        return 4215*Y+1040*(1-Y)

def get_mu(T,P):
    mu_air=1.716*10**(-5)*(T/273)**(1.5)*(273+111)/(T+111)
    mu_vapor=1.12*10**(-5)*(T/350)**(1.5)*(350+1064)/(T+1064)
    Y=get_Y(get_sat_press(T,P),P)
    return mu_vapor*Y + mu_air*(1-Y)
#https://doc.comsol.com/5.5/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.27.html#:~:text=Sutherland%E2%80%99s%20law%2C%20or%20Sutherland%E2%80%99s%20formula%2C%20is%20an%20approximation,constant.%20Each%20gas%20has%20its%20own%20Sutherland%20constant.

def get_k(T,P):
    k_air=0.0241*(T/273)**(1.5)*(273+194)/(T+194)
    k_vapor=0.0181*(T/300)**(1.5)*(350+2200)/(T+2200)
    Y=get_Y(get_sat_press(T,P),P)
    return k_vapor*Y + k_air*(1-Y)
#https://doc.comsol.com/5.5/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.27.html#:~:text=Sutherland%E2%80%99s%20law%2C%20or%20Sutherland%E2%80%99s%20formula%2C%20is%20an%20approximation,constant.%20Each%20gas%20has%20its%20own%20Sutherland%20constant.

def get_Y(Ps,P):
   return (Ps*water_mol_weight)/(Ps*water_mol_weight+(P-Ps)*air_mol_weight)

def get_spalding_mass(s,inf):
    return (inf-s)/(s-1)

def get_spalding_temp(Cp,T_inf,T_s,LH,mdot):
    return (Cp*(T_inf-T_s))/(LH)

def get_Fm(B):
    return (1+B)**(0.7)*np.log(1+B)/B

def get_Ft(B):
    return (1/(1-B))**(0.7)*np.log(1/(1-B))/B

def get_latent_h(T):
    lat_heat=2500.8-2.36*(T-273)+0.0016*(T-273)**(2)-0.00006*(T-273)**(3)
    lat_heat=lat_heat*10**3
    return lat_heat
