#Final Problem 1 - Jack James


import math
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
from scipy.integrate import odeint

#Analytically predicted steady state solution: n = C*exp(-fx/BD) where my B = Xi(?) as given.
#Given FPE: dn/dt = D(d2n/dx2)+(1/B)(d/dx)(n*dU/dx) where U=-fx

C = 1  #initial condition for n. Letting it be 1 for simplicity
f = 1
B = 1 
kbT = 300*1.3807*10**-23 #Boltzmann's Constant * 300 Kelvin
T = 1000
D = kbT/B
x0 = 0   #initial condition for x
t0 = 0   #initial condition for t
dt = 0.01
niter = T/dt

#* Initialize parameters (time step, grid spacing, etc.).
#tau = float(input('Enter time step: '))
#N = int(input('Enter the number of grid points: '))
#N=N+2  #virtual points on either side

#L = 1.        # The system extends from x=-L/2 to x=L/2
#h = L/(N-1)   # Grid size
#kappa = 1.    # Diffusion coefficient
#coeff = kappa*tau/h**2
#if coeff < 0.5 :
#    print('Solution is expected to be stable')
#else:
#    print('WARNING: Solution is expected to be unstable')

def f(u,x):
    return (u[1],u[1])
y0 = [0,0]
xs = np.linspace(0,T,1000)
us = odeint(f,y0,xs)
ys = us[:,0]


print('This was working before but now I have solid horizontal line at n=0...ctr-Z unable to save me. It is now 10 AM and I am exahausted.')
print('Still have not figured out 3d&e and now my brain is too fried to fix whatever I neglected to save for number 1.')

n = np.zeros(int(niter))
n[0] = C
nsum = C
for niter in range(T):
    n[niter] = nsum + ys[niter]
    nsum += n[niter]

plt.plot(xs,ys,'-')
plt.plot(xs,ys,'r*')
plt.xlabel('x values')
plt.ylabel('n vales')
plt.title('dn/dt = 0 = D(d2n/dx2)+(1/B)(d/dx)(n*dU/dx)')
plt.show()


#* Set initial and boundary conditions.
#Placeholder
## The boundary conditions are dn[1/2]/dx = dn[-1/2]/dx = 0

#* Set up loop and plot variables.
#xplot = np.arange(N-2)*h - L/2+h    # Record the x scale for plots
#iplot = 0                        # Counter used to count plots
#nstep = 300                      # Maximum number of iterations
#nplots = 100                     # Number of snapshots (plots) to take
#plot_step = nstep/nplots         # Number of time steps between plots


#n0 = C*np.exp((-f*x0)/(B*D))


#RK4
#def rk4(x,t,tau,derivsRK,A,B,C): 
#
#    half_tau = 0.5*tau
#    F1 = derivsRK(x,t,)  
#    t_half = t + half_tau
#    xtemp = x + half_tau*F1
#    F2 = derivsRK(xtemp,t_half,)  
#    xtemp = x + half_tau*F2
#    F3 = derivsRK(xtemp,t_half,)
#    t_full = t + tau
#    xtemp = x + tau*F3
#    F4 = derivsRK(xtemp,t_full,)
#    xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))
#    return xout	

#Derivative functions
#def derivative(X,t,):
#	return 1 #need to adapt this derivative function for later

#Comparison from your solution to midterm problem 5
#def derivMorisson(X,t,param):   
#	npart=int(len(X)/4);
#	Y=np.zeros(len(X))
#	for i in range(len(X)):
#		if(i%2==0):
#			Y[i]=X[i+1];  # dx_i/dt = v_i 
#	for i in range(npart):  # is is the particle index we're computing
#		x=X[4*i];
#		y=X[4*i+2];
#		for j in range(npart):  #j is the particle index we're computing force from
#			xx=X[4*j];
#			yy=X[4*j+2];
#			if(i!=j):  #ignoring self forces, we need to compute -r/r^3
#				sep=(x-xx)**2+(y-yy)**2;
#				sep=np.sqrt(sep);  #this is |r_i-r_j|
#				sep=sep**3;  #this is |r_i-r_j|^3
#				Y[4*i+1]-=(x-xx)/sep;
#				Y[4*i+3]-=(y-yy)/sep;  #force updates
#	return Y				
#forward time centered diferrential 

#X=np.array(x0,dtype=float)


