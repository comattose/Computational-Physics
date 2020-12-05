#HW7.3a - Jack James


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
from scipy.integrate import odeint 
import math

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



def diffequ(x, t, E):
    return np.array([x[1], -E*x[0]])

def diffequ2(x, t, EpdE):
    return np.array([x[1], -EpdE*x[0]])


E = 0.01 #energy
dE = 0.000001 #step
y0 = [0,1] #IC for system
bE = 1 
t = np.linspace(0, np.pi + (np.pi/1000000), 1000000) 

while bE > .0000001:
    sol = odeint(diffequ, y0, t, args=(E,)) #solves the differintial equation using rk4 but not the one above because my derivative is messed up
    bE = sol[999999][1] 

    EpdE = E + dE

    sol = odeint(diffequ2, y0, t, args=(EpdE,)) #solves the 2differintial equation using rk4
    bEpdE = sol[999999][1] #the value of x at time = pi

    bprime = (bEpdE - bE)/dE #defines bprime

    E = E - (bE/bprime) #updates value of E

print('part a:',E)

#second part
def diffequ(x, t, E):
    return np.array([x[1], -E*math.pow(x[0],(1/3))])

def diffequ2(x, t, EpdE):
    return np.array([x[1], -EpdE*math.pow(x[0],(1/3))])


E = 0.01 #initializes energy
dE = 0.000001 #sets energy step
y0 = [0,1] #Sets inital conditions of system
bE = 1 #initializes bE
t = np.linspace(0, np.pi + (np.pi/1000), 1000) #sets the time interval and delta t

while bE > .00001:
    sol = odeint(diffequ, y0, t, args=(E,)) #solves the differintial equation using rk4
    bE = sol[999][1] #the value of x at time = pi

    EpdE = E + dE

    sol = odeint(diffequ2, y0, t, args=(EpdE,)) #solves the differintial equation using rk4
    bEpdE = sol[999][1] #the value of x at time = pi

    bprime = (bEpdE - bE)/dE #defines bprime

    E = E - (bE/bprime) #updates value of E

print('part b',E)