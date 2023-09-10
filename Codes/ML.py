from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Data/k.csv')
df['$k$ [m/s]'] = df['$k$ [m/s]'].fillna(0)
df.loc[df['$k$ [m/s]'] > 0, '$k$ [m/s]'] = 1
y=df.values[:,-1]
y=y.astype('int')

X=np.array([df['Ambient Pressure [Pa]'],df['dT [K]']]).T

#Get your SVM set up using the SVC library.
x1line = np.linspace(101300,401300, 200)
x2line = np.linspace(1, 40, 200) 
near=10
neigh=KNeighborsClassifier(n_neighbors=near)
neigh.fit(X,y)
x1grid, x2grid = np.meshgrid(x1line, x2line)
Xgrid = np.array([x1grid, x2grid]).reshape([2,200**2]).T
y_kn=neigh.predict(Xgrid)
fig=plt.contourf(x1line,x2line,y_kn.reshape([200,200]))
plt.xlabel('Pressure [Pa]')
plt.ylabel('dT [K]')
plt.title('Found Solution (1=Found, 0=No Solution)')





plt.colorbar()
# plt.scatter(X[y==1][:,0],X[y==1][:,1])
# plt.scatter(X[y==0][:,0],X[y==0][:,1])
plt.savefig('Plots/Solutions_diagram_KNeighbor.pdf')
plt.show()