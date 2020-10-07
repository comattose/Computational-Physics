#HW4.1 Jack James

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy 
import sympy as sym

def rk45(x, t0, y0, t_bound):      # not sure I'll use this yet. Leaning towards RK4 since have dr/dt and r. Only creating function so I remember.
    scipy.integrate.rk45(x, t0, y0, t_bound)

def rk4(x,t,tau,derivsRK,phi): # still couldn't get it to import right so I just copy pasted.
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
    F1 = derivsRK(x,t)  
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp,t_half)  
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp,t_half)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp,t_full)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
    return xout		
 
plt.axes(projection = 'polar')    # setting test axes projection as polar

def derivTest(a,r):
    return r*(a-r**2)


t = 0  
t_max = 5
a = 0.81   # creating an a that's a square to make sure I have everything set up right. using 0.81 and 1.21. can't seem to get label for where inner ring is
r = 2.0    # setting generic radius 
phi = 0.0  
phi_max = -5.0 * np.pi

dphi = -1
dt = 0.05
#drdt = r*(a-r**2)
dphidt = -1

niter = t_max / dt



P=np.array([0.0,0.0],dtype=float)  # creating array to store data for rk4
rk4Result=np.asarray([r])
phiResult=np.asarray([phi])

P[0]=r
P[1]=r*(a-r**2)
t=0



for titer in range(int(niter)):
    P=rk4(P,t,dt,derivTest,phi,)
    rk4Result=np.append(rk4Result,[P[0]])  #store the value of theta we saw
    t=t+dt
    phi = phi + dphi
    phiResult = np.append(phiResult, phi)




#rads = np.arange(0, phi_max, 0.1)     #test array containing the radian values
  

for i in range(int(niter)):        # plotting the circle
    plt.polar(phiResult[i], rk4Result[i], 'g.') 
  

#plt.show()       # display the polar plot. This one doesn't work.


mp.pyplot.polar(phiResult, rk4Result, [], 'g.')    # display polar plot

print("Part a is included as a pdf.\nI am having trouble getting graph to display when goes to origin/shows circle after changing my 'a' value.")