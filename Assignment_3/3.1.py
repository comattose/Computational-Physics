#HW3.1 Jack James

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

										
def deriv(X,t,param):
	return np.array([X[1],-np.sin(X[0])])  # this is dX/dt = deriv(X,t)
											#note that this funny form is chosen to match Garcia's rk4 function
							
										

def euler(X,t,tau,derivs,params):		#this defines the euler update.  Pretty easy to code, but remember it's innacurate
	return X+tau*deriv(X,t,params)		#the function definition matches rk4

theta=0.1*np.pi
omega=0.0
t=0.0
T=30.0
g=9.18
L=5.0
m=5.0 #just putting in an integer for mass and length since it's just a coefficient for both terms


#these are the parameters for our system


dt=0.01
niter=T/dt
#these are the parameters for our integrator

tet = (m*g*L*((0.5*(theta**2)))-1) #initial theta energy term
oet = m*(L**2)*0.5*(omega**2) #initial omega energy term = 0
#print(tet+oet) testing issues with calculation for initial energy

X=np.array([0.0,0.0],dtype=float)  #where we will store the data at each timestep for euler / rk4
sampleTimes=np.asarray(range(int(niter)+1))*dt
eulerResult=np.asarray([tet])

    #theta energy term = m*g*L*((0.5*(theta**2))-1)
    #omega energy term = m*(L**2)*0.5*(omega**2) 

X[0]=theta
X[1]=omega


t=0

for titer in range(int(niter)):
    X=euler(X,t,dt,deriv,[])
    tet=(m*g*L*((0.5*(X[0]**2)))-1) #theta energy term
    oet=(m*(L**2)*0.5*(X[1]**2)) #omega energy term
    eulerResult=np.append(eulerResult,[(tet+oet)])  #store the value of theta we saw
    #print('theta: ', tet, '\t', 'omega: ', oet, 'energy: ', tet+oet)
    t=t+dt		
    

fig,(a1)=plt.subplots(1,1)

a1.plot(sampleTimes,eulerResult/2/np.pi,color='b',label='Euler')

ax1=a1.axes

a1.legend(loc='upper right')
a1.set_xlabel('Time')
a1.set_ylabel('Energy of Pendulum')
plt.tight_layout()
plt.show()	
