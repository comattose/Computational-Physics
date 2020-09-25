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
							
										

#def euler(X,t,tau,derivs,params):		#this defines the euler update.  Pretty easy to code, but remember it's innacurate
#	return X+tau*deriv(X,t,params)		#the function definition matches rk4






X=0
V=1
t=0.0
T=5.0
#these are the parameters for our system


dt=0.5
niter=T/dt
#these are the parameters for our integrator


P=np.array([0.0,0.0],dtype=float)  #where we will store the data at each timestep for euler / rk4
sampleTimes=np.asarray(range(int(niter)+1))*dt
rk4Result=np.asarray([X])		#where we will store the data for plotting
sinResult=np.asarray(np.sin([X]))   #creating sin function to compare
verletResult=np.asarray([X])


P[0]=X
P[1]=V
t=0

for titer in range(int(niter)):
	P=rk4(P,t,dt,deriv,[])
	rk4Result=np.append(rk4Result,[P[0]])  #store the value of theta we saw
	t=t+dt
print("For dt =", dt, ": RK4 X(5) =", P[0])	


t=0
for titer in range(int(niter)):
    sinResult=np.append(sinResult,[np.sin(t+dt)]) #creating sin values
    t=t+dt
		

	
X=0
V=1
t=0
P[0]=X  												#initial value of the angle
P[1]=-(X+V*dt+dt*dt*(-np.sin(X))/2)					#next value of the angle (startup)
np.append(verletResult,[P[1]])
np.append(verletResult,[P[0]])
for titer in range(int(niter)):
	pcurr=P[0]									#current value of x
	P[0]=(2*P[0]-P[1]+dt*dt*(-np.sin(pcurr)))		#update the value of the current x
	P[1]=pcurr									#update the value of the old x
	verletResult=np.append(verletResult,[P[0]])  			#store the value of theta we saw
	t=t+dt
print("\t\t\t\t\t\t\t\t\t\t\t\t\t\tand Verlet X(5) =", P[0])


fig,(a1)=plt.subplots(1,1)

a1.plot(sampleTimes,(sinResult-rk4Result)/2/np.pi,color='r',label='Sin - rk4')
a1.plot(sampleTimes,(sinResult-verletResult)/2/np.pi,color='k',label='Sin - Verlet')

ax1=a1.axes

a1.legend(loc='lower left')
a1.set_xlabel('t with dt=0.5')
a1.set_ylabel('theta/2pi')

plt.tight_layout()
plt.show()	



#Creating same functions without the appends to print out values for each dt.
#This section is for dt=0.1
X=0
V=1
t=0
T=5.0
dt=0.1
niter=T/dt

P=np.array([0.0,0.0],dtype=float)  

P[0]=X
P[1]=V
for titer in range(int(niter)):
	P=rk4(P,t,dt,deriv,[])  
	t=t+dt
print("For dt =", dt, ": RK4 X(5) = ", P[0])		
	
X=0
V=1
t=0
P[0]=X  												
P[1]=-(X+V*dt+dt*dt*(-np.sin(X))/2)					
for titer in range(int(niter)):
	pcurr=P[0]									
	P[0]=(2*P[0]-P[1]+dt*dt*(-np.sin(pcurr)))		
	P[1]=pcurr									
	t=t+dt
print("\t\t\t\t\t\t\t\t\t\t\t\t\t\tand Verlet X(5) =", P[0])

#dt=0.05
X=0
V=1
t=0
T=5.0
dt=0.05
niter=T/dt

P=np.array([0.0,0.0],dtype=float)  

P[0]=X
P[1]=V
for titer in range(int(niter)):
	P=rk4(P,t,dt,deriv,[])  
	t=t+dt
print("For dt =", dt, ": RK4 X(5) =", P[0])		
	
X=0
V=1
t=0
P[0]=X  												
P[1]=-(X+V*dt+dt*dt*(-np.sin(X))/2)					
for titer in range(int(niter)):
	pcurr=P[0]									
	P[0]=(2*P[0]-P[1]+dt*dt*(-np.sin(pcurr)))		
	P[1]=pcurr									
	t=t+dt
print("\t\t\t\t\t\t\t\t\t\t\t\t\t\tand Verlet X(5) =", P[0])

#dt=0.01
X=0
V=1
t=0
T=5.0
dt=0.01
niter=T/dt

P=np.array([0.0,0.0],dtype=float)  

P[0]=X
P[1]=V
for titer in range(int(niter)):
	P=rk4(P,t,dt,deriv,[])  
	t=t+dt
print("For dt =", dt, ": RK4 X(5) =", P[0])		
	
X=0
V=1
t=0
P[0]=X  												
P[1]=-(X+V*dt+dt*dt*(-np.sin(X))/2)					
for titer in range(int(niter)):
	pcurr=P[0]									
	P[0]=(2*P[0]-P[1]+dt*dt*(-np.sin(pcurr)))		
	P[1]=pcurr									
	t=t+dt
print("\t\t\t\t\t\t\t\t\t\t\t\t\t\tand Verlet X(5) =", P[0])

#dt=0.005
X=0
V=1
t=0
T=5.0
dt=0.005
niter=T/dt

P=np.array([0.0,0.0],dtype=float)  

P[0]=X
P[1]=V
for titer in range(int(niter)):
	P=rk4(P,t,dt,deriv,[])  
	t=t+dt
print("For dt =", dt, ": RK4 X(5) =", P[0])		
	
X=0
V=1
t=0
P[0]=X  												
P[1]=-(X+V*dt+dt*dt*(-np.sin(X))/2)					
for titer in range(int(niter)):
	pcurr=P[0]									
	P[0]=(2*P[0]-P[1]+dt*dt*(-np.sin(pcurr)))		
	P[1]=pcurr									
	t=t+dt
print("\t\t\t\t\t\t\t\t\t\t\t\t\t\tand Verlet X(5) =", P[0])

#dt=0.001
X=0
V=1
t=0
T=5.0
dt=0.001
niter=T/dt

P=np.array([0.0,0.0],dtype=float)  

P[0]=X
P[1]=V
for titer in range(int(niter)):
	P=rk4(P,t,dt,deriv,[])  
	t=t+dt
print("For dt =", dt, ": RK4 X(5) =", P[0])		
	
X=0
V=1
t=0
P[0]=X  												
P[1]=-(X+V*dt+dt*dt*(-np.sin(X))/2)					
for titer in range(int(niter)):
	pcurr=P[0]									
	P[0]=(2*P[0]-P[1]+dt*dt*(-np.sin(pcurr)))		
	P[1]=pcurr									
	t=t+dt
print("\t\t\t\t\t\t\t\t\t\t\t\t\t\tand Verlet X(5) =", P[0])