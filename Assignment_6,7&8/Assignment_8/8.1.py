#Assignment 8 - Problem 1 - Jack James

import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import csv
import sys
from sympy import symbols, diff 
from scipy.fft import fft, ifft
import random
from numpy.random import random_sample
import scipy.integrate as integrate

#a = 7**5
#b = 0
#m = 2**31-1
#X = (a*x+b)*mod(m)
#T = random.randint(2,50) #some T
#print('T =',T)

#T = 5000 #some T

#P = np.zeros(T) #empty array for discrete probabilities
#A = np.zeros(T) #empty array for sum of each 'chunk' of probabilities
#m = 0 #placeholder to add each subsequent 'chunk' to A array


#for i in range(1,T):
#    P[i] = i**-1
#    m += P[i]
#    A[i] = m

#print('m =',m)    
#print('A =',A)

#r = random.random()/m             
#i = 0
#found = 0

#while found==0:
#    if(A[i]>r):
#        C = i
#        found = 1
#    if(i==T-1):
#        C = i
#        found = 1
#    i += 1
    
#print('Random integer:',C)
print('For a random discrete integer outcome with weight p(x)=x^-1:')
def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)                     #This creates the bins 
    return values[np.digitize(random_sample(size), bins)]       #This sees which bin the number falls into and returns the corresponding value

T = 500  #Some number T
values = np.zeros(T) #Creating empty array to store index values since python starts at zero.
probabilities = np.zeros(T) #Creating empty array to store probabilities x^-1
total = 0
for i in range(1,T):
    values[i] = i+1
    probabilities[i]=(i+1)**-1
    total += probabilities[i]  #Placeholder variable to normalize probabilities afterwards
    
#print(total)    
#print(values[T-1])  #test
#print(probabilities) #test

#probabilities[0] = 1     #For whatever reason, this indexed value was showing up as zero before so I'm just manually making it 1 here. Fixed?
probabilities[:] = [x/total for x in probabilities] #Here I divide each value by total sum of probabilities to normalize each probability. 


print('(Between 1 and 500)',weighted_values(values, probabilities, 1))

#print('For a random continuous outcome with weight p(x)=x^-1:')

#integral = integrate.quad(lambda x: x**-1,1,T)
#print(integrate.quad(lambda x: x**-1,1,T))
#r = random.uniform(0,1)

#I think I see how to create expected distribution from this but not how to generate 10^5 numbers and plot that distribution.
#You just have to set r = integral and then solve in terms of x.

#For T = 10, generate 10^5 numbers using this program and plot the distribution
print('For T = 10, distribution is as follows:')

#def f():
#    T = 10
#    values = np.zeros(T) #Setting this up as a function since we're doing it 10^5 times.
#    probabilities = np.zeros(T)
#    total = 0
#    
#    for i in range(1,T):
#        values[i] = i+1
#        probabilities[i]=(i+1)**-1
#        total += probabilities[i]
#        
#    probabilities[0] = 1
#    probabilities[:] = [x/total for x in probabilities]
#    return(weighted_values(values, probabilities, 1))

T = 10
values = np.zeros(T) #Setting this up as a function since we're doing it 10^5 times.
probabilities = np.zeros(T)
total = 0
for i in range(1,T):
    values[i] = i+1
    probabilities[i]=(i+1)**-1
    total += probabilities[i]
#probabilities[0] = 1
probabilities[:] = [x/total for x in probabilities]
#print(probabilities)
#probabilities[0]=1/total
result = weighted_values(values, probabilities, 10**5)


#print(result)    
#plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
plt.hist(result, bins=20) #different bin
plt.gca().set(title='T = 10', ylabel='Frequency');