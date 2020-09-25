#HW3.2 Jack James

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

print('Part A is included as a PDF\n')
									
def rk4(x,t,tau,derivsRK,param): #couldn't get it to import right so I just copy pasted.
    """Runge-Kutta integrator (4th order)
       Input arguments -
        x = current value of dependent variable
        t = independent variable (usually time)
        tau = step size (usually timestep)
        derivsRK = right hand side of the ODE; derivsRK is the
                  name of the function which returns dx/dt
                  Calling format derivsRK (x,t,param).
        param = extra parameters passed to derivsRK
       Output arguments -
        xout = new value of x after a step of size tau
    """
    
    half_tau = 0.5*tau
    F1 = derivsRK(x,t,param)  
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half,param)  
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half,param)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full,param)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout		
										
								
										
def deriv(X,t,param):
	return np.array([X[1],-np.sin(X[0])])  # this is dX/dt = deriv(X,t)
											#note that this funny form is chosen to match Garcia's rk4 function
							
										




X=0.5
V=0
A=-0.25
J=0 #assuming initial jerk is zero
z=0.0
Z=5.0


dz=0.1
niter=Z/dz


P=np.array([0.0,0.0],dtype=float)
Q=np.array([0.0,0.0],dtype=float) #creating two more coupled pair
T=np.array([0.0,0.0],dtype=float)
sampleTimes=np.asarray(range(int(niter)+1))*dz
rk4Result=np.asarray([X])
sechResult=np.asarray([X])   #creating sech function to compare


P[0]=X
P[1]=V

Q[0]=V
Q[1]=A

T[0]=A
T[1]=J

for titer in range(int(niter)):
    T=rk4(T,z,dz,deriv,[])
    Q=rk4(Q,z,dz,deriv,[])
    P=rk4(P,z,dz,deriv,[])
    solution=(1-(T[0]/Q[0]))/6
    Q[1]=T[0]                                 #updating values for each derivative for the next cycle
    P[1]=Q[0]
    rk4Result=np.append(rk4Result,solution)  
    #rk4Result=np.append(rk4Result,[P[0]])    #trying to figure out how to get formula to work
    SR=[0.5*(np.cosh(z/2))**-2]
    sechResult=np.append(sechResult,SR)
    z=z+dz
print("The value of w(5)=", P[0], "and the value of exact solution is", SR)	


fig,(a1)=plt.subplots(1,1)

a1.plot(sampleTimes,(rk4Result)/2/np.pi,color='r',label='rk4')
a1.plot(sampleTimes,(sechResult)/2/np.pi,color='k',label='0.5*sech^2(z/2)')

ax1=a1.axes

a1.legend(loc='lower left')
a1.set_xlabel('t with dt=0.5')
a1.set_ylabel('theta/2pi')

plt.tight_layout()
plt.show()	