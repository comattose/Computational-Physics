#HW4.2 Jack James

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy 
import sympy as sym
from numpy import linalg as LA

mat = np.zeros((50,50))    #creating empty matrix
b0 = np.zeros((50))        #creating empty vector 
print("mat is ", mat)

for i in range(50):        #filling requested values
    for j in range(50):
        if abs(i-j)<3:
            mat[i,j]=1
            b0[i]=1
print("b0 = ", b0)
EigenValues, EigenVectors=LA.eig(mat)
print("2C: The last eigenvalue is ", EigenValues[-1], "with eigenvector ", EigenVectors[-1],".\n")

largest_mag = 0
location = 0
for i in range(50):
    mag = 0
    for j in range(50):
        for k in range(50):
            square = float(EigenVectors[j,k]**2)
            mag = mag + square
            if largest_mag < mag:
                largest_mag = mag
                location = k

print(" The largest eigenvalue is ", max(EigenValues), "and the largest eigenvector is ", EigenVectors[location], "with magnitude", largest_mag, ".")

bk=np.asarray([3])

for i in range(49):
    bk = np.append(bk,3)
def res(test_list): 
    return sum(map(lambda i : i * i, test_list)) 
print(bk)
for i in range(500):
    bk = mat * bk / (res(bk))
    a = res(bk) * 10**-6
    b = res(bk)-res(b0)      #been trying to get this if statement to work a million ways. I don't understand why it gives error about truth value
    if b > a:                #These are just NUMBERS...
        continue
print("bkmax is", bk, ", M * bk max =", mat*bk, " and lambda max * bk max =", max(EigenValues)*bk)