import numpy as np
import matplotlib.pyplot as plt

plt.figure()
# d=0.9*10**(-10)
# for delta in range(10,12):


d=11**(-10)
kb=1.380649*10**(-23)
mol_mass=0.018
rhol=958.77
rhov=0.5977
drop_rad=1*10**(-6)

dsmol=85
p=101325
T=np.arange(368,373,1)
S=p/(p*np.exp(-(2256500/461.4)*((1.0/T)-(1/373))))
S2=p/(p*np.exp(-(2256500/461.4)*((1.0/T)-(1/373))))+1
# r_crit[cellI]=(2*sigma.value())/(rhol_kg.value()*R.value()*T[cellI]*std::log(supersat[cellI]));
r=2*0.0588/(rhol*461.4*T*np.log(S))
r2=2*0.0588/(rhol*461.4*T*np.log(S))

Nmol=p/kb/T  

# calculation


dg=-kb*T*np.log(S2)*Nmol
dhmol=8.31447215*(np.log(S2))/(-(1/T)+(1/373))
etha=dg/(dhmol)
# print(etha)
q=np.sqrt(1-etha)
psii = 2 * (1 + q) * etha**(-3) - (3 + 2 * q) * etha**(-2) + etha**(-1)
W = -(4 * np.pi / 3) * d**(3) * dg * psii
Rs=d*(1+q)*etha**(-1)
A_star=4*np.pi*Rs**(2)
lamb=p*(2*np.pi*mol_mass*kb*T)
Z_star=np.sqrt(np.abs(dg)*(Rs-d/etha)/(kb*T))*(1/(Nmol*2*np.pi*Rs**(2)))
J = Nmol*A_star*lamb*Z_star
I = J * np.exp(-W / (T * kb))
#CNT


# plt.plot(T,r)
# plt.plot(T,r2)
# plt.plot(T,Rs)
plt.plot(T,I)
plt.show()

