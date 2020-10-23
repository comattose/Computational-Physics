#Midterm Problem #4 (python) - Jack James


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


#ln(A)=-summation((I-A)^k/k,k=1,infinity)

seed(rand(-10,10))
N = 10
max = 80                            #Maximum k we will execute. This value is large enough that precision isn't improved by increasing it
                                    #since only 9 digits are displayed. 
M = np.zeros((N,N))
for i in range(N):                  #loop to create random matrix integer values from -10 to 10
    for j in range(N):
        M[i,j] = rand(-10,10)
#print("M is",M)

A = np.zeros((N,N))
for i in range(N):                  #loop to create requested Aij=(1 + Kroniker Delta ij)/(n+1)
    for j in range(N):
        if i == j:
            A[i,j] = 2/(N+1)
        else:
            A[i,j] = 1/(N+1)
#print("A is",A)

I = np.zeros((N,N))
for i in range(N):                  #loop to create identity matrix
    for j in range(N):
        if i == j:
            I[i,j] = 1
#print("I is",I)            
            

def log_mat(Mat,N,max):                 #creating function that sums up taylor series given
    Ozzy = np.zeros((N,N))              #Ozzy is just empty matrix we will add each series term to. 
    for k in range(1,max+1):
        Ozzy -= (np.linalg.matrix_power((I-Mat),k)/k)       #This was the issue. Using ** operator on matrix didn't do what I thought it did.
        #Ozzy -= ((I-Mat)**k)/k           #getting different values depending on language used so added step-by-step version to make sure I didn't make mistake.
        #Rhoads = I-Mat                    
        #Kerslake = Rhoads**k
        #Cook = Kerslake/k
        #Ozzy = Ozzy - Cook
        #print(k,"term",Ozzy)
    return Ozzy   

#print("console\n")
print(log_mat(A,N,max))

original_stdout = sys.stdout

with open('C:/Users/Jack/Documents/Fall 2020/Comp Phys/Assignments/Midterm/M.4.output.txt', 'w') as f:    #exporting to file
    sys.stdout = f
    #print("file")
    print(log_mat(A,N,max))
    sys.stdout = original_stdout
    