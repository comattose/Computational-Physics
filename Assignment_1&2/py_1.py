
import numpy as np 
import math
import matplotlib.pyplot as plt

t=np.zeros(20) #create time array with 20 empty cells
S=np.zeros(20) #equation in question placeholder
#print(S) test

def X(z): #create function x
    return(1/(1+(9*np.exp(-z))))

def FtoC(j,k): #create fundamental theorem of calculus function
    return(-(X(j-k)+X(j))/k)

from scipy.misc import derivative #central difference formula

for i in range(len(t)):    
    t[i]=(10**(-i)) #create time values for log scale
    a=FtoC(i,t[i]) #trying to make sure FtoC and deriv return scalar
    b=derivative(X,t[i],10**-30,1) #using function X, calculates n=1st derivative at t[i] with dt=10^-30
    S[i]=abs(b-a) #place difference in respective cell
#    print(c, " line ", i) testing issues with my formulas
    
#print(S)
    
plt.loglog(t,S)
plt.ylabel('Delta = dx(t)/dt - FToC')
plt.xlabel('Time (s)')