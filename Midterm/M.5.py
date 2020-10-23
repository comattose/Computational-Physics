#Midterm Problem #5 - Jack James


import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import csv
import sys
from sympy import symbols, diff 
from scipy.fft import fft, ifft
from random import seed
from random import randint as rand
from random import random as random

print("C: Sure, there was a periodic orbit. I guess that indicates the instability?")
print("D: I had no difference, maybe my dt was already small enough?")

# Part C

def rk4(x,t,tau,derivsRK,A,B,C): 

    half_tau = 0.5*tau
    F1 = derivsRK(x,t,)  
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half,)  
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half,)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full,)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout	



#Derivative functions
def derivative(X,t,):
	return np.array([X[1],-np.sin(X[0])])
def derivative2(A,B,C):
    return np.array(d2rdt2(A,B,C))


#Variables	
t = 0
T = 50
e = 10**-6
dt = 0.05
niter = T/dt


def d2rdt2(A,B,C):                  #creating function to calculate acceleration at each step
    return np.array([(-(A[0]-B[0])**-2)-((A[0]-C[0])**-2),(-(A[1]-B[1])**-2)-((A[1]-C[1])**-2)])


R1=np.array([0.0,0.0],dtype=float)  #creating array to store data for position & velocity
R2=np.array([0.0,1.0],dtype=float)
R3=np.array([0.0,-1.0],dtype=float)
V1=np.array([0.0,0.0],dtype=float)
V2=np.array([1.0,0.0],dtype=float)
V3=np.array([-1.0,0.0],dtype=float)

A1=np.array([0.0,0.0],dtype=float)  #acceleration
A2=np.array([0.0,7/8],dtype=float)  #solved for this in OneNote pdf
A3=np.array([0.0,1/4],dtype=float)


#Energies
def Kin(A,B,C):                     #kinetic equation formula for each step
    return np.array([0.5*(A[0]**2+B[0]**2+C[0]**2)+0.5*(A[1]**2+B[1]**2+C[1]**2)],dtype=float)
def Pot(A,B):                       #potential equation formula for each step
    return np.array([(1/abs((((A[0]-B[0])**2)+((A[1]-B[1])**2))**0.5))],dtype=float)

K=Kin(V1,V2,V3)                     #initial kinetic
P=Pot(R1,R2)+Pot(R2,R3)+Pot(R1,R3)  #initial potential
E=K+P                               #initial total energy


#print("This is the initial K",K,", P,",P,", and E",E)  #FIXED potential energy having inf issues

for i in range(int(niter)):

    V1=rk4(R1,t,dt,derivative,R1,R2,R3,)
    V2=rk4(R2,t,dt,derivative,R1,R2,R3,)
    V2=rk4(R3,t,dt,derivative,R1,R2,R3,)
    
    R1=R1+V1*dt
    R2=R2+V2*dt
    R3=R3+V3*dt
 
    t=t+dt
    
    K=np.append(K,[Kin(V1,V2,V3)])
    P=np.append(P,[Pot(R1,R2)+Pot(R2,R3)+Pot(R1,R3)])
    E=np.append(E,[K[i]+P[i]])
#print("This is the final K",K,", P,",P,", and E",E)

#Plot
sampleTimes=np.asarray(range(int(niter)+1))*dt
fig,(a1)=plt.subplots(1,1)

a1.plot(sampleTimes,(K)/2/np.pi,color='r',label='Kinetic')
a1.plot(sampleTimes,(P)/2/np.pi,color='k',label='Potential')
a1.plot(sampleTimes,(E)/2/np.pi,color='k',label='Total')

ax1=a1.axes

a1.legend(loc='lower left')
a1.set_xlabel('Time')
a1.set_ylabel('Energy')

plt.tight_layout()
plt.show()	

#Part D

#Variables	
t = 0
T = 50
e = 10**-6
dt = 0.05
niter = T/dt

R1=np.array([0.01,0.0],dtype=float)  #creating array to store data for position & velocity
R2=np.array([0.0,1.0],dtype=float)
R3=np.array([0.0,-1.0],dtype=float)
V1=np.array([0.0,0.0],dtype=float)
V2=np.array([1.0,0.0],dtype=float)
V3=np.array([-1.0,0.0],dtype=float)

A1=np.array([0.0,0.0],dtype=float)  #initial accelerations in case I need it
A2=np.array([0.0,7/8],dtype=float)  #solved for this in OneNote pdf
A3=np.array([0.0,1/4],dtype=float)

K=Kin(V1,V2,V3)                     #initial kinetic
P=Pot(R1,R2)+Pot(R2,R3)+Pot(R1,R3)  #initial potential
E=K+P                               #initial total energy

#print("This is the initial K",K,", P,",P,", and E",E)  #FIXED potential energy having inf issues

for i in range(int(niter)):

    V1=rk4(R1,t,dt,derivative,R1,R2,R3,)
    V2=rk4(R2,t,dt,derivative,R1,R2,R3,)
    V2=rk4(R3,t,dt,derivative,R1,R2,R3,)
    
    R1=R1+V1*dt
    R2=R2+V2*dt
    R3=R3+V3*dt
 
    t=t+dt
    
    K=np.append(K,Kin(V1,V2,V3))
    P=np.append(P,Pot(R1,R2)+Pot(R2,R3)+Pot(R1,R3))
    E=K+P
#print("This is the final K",K,", P,",P,", and E",E)

#Plot
sampleTimes=np.asarray(range(int(niter)+1))*dt
fig,(a1)=plt.subplots(1,1)

a1.plot(sampleTimes,(K)/2/np.pi,color='r',label='Kinetic')
a1.plot(sampleTimes,(P)/2/np.pi,color='k',label='Potential')
a1.plot(sampleTimes,(E)/2/np.pi,color='k',label='Total')

ax1=a1.axes

a1.legend(loc='lower left')
a1.set_xlabel('Time')
a1.set_ylabel('Energy')

plt.tight_layout()
plt.show()	


#Part D

#Variables	
t = 0
T = 50
e = 10**-6
dt = 0.005
niter = T/dt

R1=np.array([0.01,0.0],dtype=float)  #creating array to store data for position & velocity
R2=np.array([0.0,1.0],dtype=float)
R3=np.array([0.0,-1.0],dtype=float)
V1=np.array([0.0,0.0],dtype=float)
V2=np.array([1.0,0.0],dtype=float)
V3=np.array([-1.0,0.0],dtype=float)

A1=np.array([0.0,0.0],dtype=float)  #initial accelerations in case I need it
A2=np.array([0.0,7/8],dtype=float)  #solved for this in OneNote pdf
A3=np.array([0.0,1/4],dtype=float)

K=Kin(V1,V2,V3)                     #initial kinetic
P=Pot(R1,R2)+Pot(R2,R3)+Pot(R1,R3)  #initial potential
E=K+P                               #initial total energy

#print("This is the initial K",K,", P,",P,", and E",E)  #FIXED potential energy having inf issues

for i in range(int(niter)):

    V1=rk4(R1,t,dt,derivative,R1,R2,R3,)
    V2=rk4(R2,t,dt,derivative,R1,R2,R3,)
    V2=rk4(R3,t,dt,derivative,R1,R2,R3,)
    
    R1=R1+V1*dt
    R2=R2+V2*dt
    R3=R3+V3*dt
 
    t=t+dt
    
    K=np.append(K,Kin(V1,V2,V3))
    P=np.append(P,Pot(R1,R2)+Pot(R2,R3)+Pot(R1,R3))
    E=K+P
#print("This is the final K",K,", P,",P,", and E",E)

#Plot
sampleTimes=np.asarray(range(int(niter)+1))*dt
fig,(a1)=plt.subplots(1,1)

a1.plot(sampleTimes,(K)/2/np.pi,color='r',label='Kinetic')
a1.plot(sampleTimes,(P)/2/np.pi,color='k',label='Potential')
a1.plot(sampleTimes,(E)/2/np.pi,color='k',label='Total')

ax1=a1.axes

a1.legend(loc='lower left')
a1.set_xlabel('Time')
a1.set_ylabel('Energy')

plt.tight_layout()
plt.show()	

print("C: Sure, there was a periodic orbit. I guess that indicates the instability?")
print("D: I had no difference, maybe my dt was already small enough?")