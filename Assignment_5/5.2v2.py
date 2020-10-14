# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 23:00:26 2020

@author: Jack
"""

from __future__ import division
import numpy as np
import math
import matplotlib.pylab as plt
import sympy as sp
from numpy.linalg import inv

#initial guesses
x = 1.0
y = 2.0
i1 = 0

while i1<12:
#    F = (x**2)*np.exp(-x**2)+y**2
#    G = (x**4)/(1+(x**2)(y**2))
    F= np.matrix([[((x**2)*np.exp(-x**2)+(y**2))],[((x**4)/(1+((x**2)*(y**2))))]])        #couldn't get it to work with matrices
    theta = np.sum(F)
    J = np.matrix([[-2*np.exp(-x**2)*x*(x**2-1),2*y],[(2*x**3)*((x**2)*(y**2)+2)/(((x**2)*(y**2)+1)**2),(-2*(x**6)*y)/(((x**2)*(y**2)+1)**2)]])
    Jinv = inv(J) 
    xn = np.array([[x],[y]])    
    xn_1 = xn - (Jinv*F)
    x = xn_1[0,0]
    y = xn_1[1,0]
    #~ print theta
    print(xn)
    i1 = i1+1
print("x =",x,"and y =",y )