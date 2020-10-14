#HW5.2 Jack James

#Newton's method to find any x*, y* such that F(x*,y*)=G(x*,y*)=1
#F(x,y)=(x**2)exp(-x**2)+y**2
#G(x,y)=x**4/(1+(x**2)(y**2))

#y = f'(xo) * (x-xo) + f(xo)    tangent line
#0 = f'(xo)(x-xo)+f(xo)         x-intercept
#x = xo - f(xo)/f'(xo)          first iteration

#x(n+1) = xn - f(xn)/f'(xn)     final iteration

from __future__ import division
import numpy as np
import math
import matplotlib.pylab as plt
import sympy as sp
from numpy.linalg import inv

#def newton(f,Df,x0,epsilon,max_iter):
#    xn = x0
#    for n in range(0,max_iter):
#        fxn = f(xn)
#        if abs(fxn) < epsilon:
#            print('Found solution after',n,'iterations.')
#            return xn
#        Dfxn = Df(xn)
#        if Dfxn == 0:
#            print('Zero derivative. No solution found.')
#            return None
#        xn = xn - fxn/Dfxn
#        print('Exceeded maximum iterations. No solution found.')
#        return None
    
#def variable_input(p,dpx,dpy):                  #creating way to easily input x or y
#    approx = newton(p,dpx,dpy,1**-2,100)
    
#x=1
#y=2

#initial guesses
x = -1.2
y = 0.8
i = 0

print("Tried this two ways.")

while i<6:
#    F = (x**2)*np.exp(-x**2)+y**2
#    G = (x**4)/(1+(x**2)(y**2))
    F= np.matrix([[((x**2)*np.exp(-x**2)+(y**2))],[((x**4)/(1+((x**2)*(y**2))))]])        #couldn't get it to work with matrices
    theta = np.sum(F)
    J = np.matrix([[-2*np.exp(-x**2)*x*(x**2-1),2*y],[(2*x**3)*((x**2)*(y**2)+2)/(((x**2)*(y**2)+1)**2),(-2*(x**6)*y)/(((x**2)*(y**2)+1)**2)]])
    Jinv = inv(J) 
    xyn = np.array([[x],[y]])    
    xn_1 = xyn - (Jinv*F)
    x = xn_1[0,0]
    y = xn_1[1,0]
    xyn = xn_1
    i = i+1
print("x =",x,"and y =",y )
print("theta",i+1,"is",theta)
print("final x and y are", xyn,"\n\n")


print("Attempt 2")
def F(x,y):
    F=(x**2)*np.exp(-x**2)+y**2
    return F
def dFdx(x,y):
    dFdx=-2*np.exp(-x**2)*x*(x**2-1)
    return dFdx
def dFdy(x,y):
    dFdy=2*y
    return dFdy

def G(x,y):
    G=(x**4)/(1+(x**2)*(y**2))
    return G
def dGdx(x,y):
    dGdx=(2*x**3)*((x**2)*(y**2)+2)/(((x**2)*(y**2)+1)**2)
    return dGdx
def dGdy(x,y):    
    dGdy=(-2*(x**6)*y)/(((x**2)*(y**2)+1)**2)
    return dGdy


x=-1.20
y=0.8

def newton(x,y,epsilon,max_iter):
    F_mat= np.matrix([[((x**2)*np.exp(-x**2)+(y**2))],[((x**4)/(1+((x**2)*(y**2))))]])        #having trouble getting it to work with matrices
    theta = np.sum(F)
    J = np.matrix([[-2*np.exp(-x**2)*x*(x**2-1),2*y],[(2*x**3)*((x**2)*(y**2)+2)/(((x**2)*(y**2)+1)**2),(-2*(x**6)*y)/(((x**2)*(y**2)+1)**2)]])
    Jinv = inv(J)
    xyn = np.array([[x],[y]])
    print("original x and y are",xyn)
    for n in range(0,max_iter):
        fn = F(xyn[0],xyn[1])
        #print("fn is",fn)
        gn = G(xyn[0],xyn[1])
        #print("gn is",gn)
        if abs(fn) < epsilon and abs(gn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xyn
        Dfxn = dFdx(xyn[0],xyn[1])
        Dfyn = dFdy(xyn[0],xyn[1])
        Dgxn = dGdx(xyn[0],xyn[1])
        Dgyn = dGdy(xyn[0],xyn[1])
        if Dfxn == 0 or Dgxn == 0 or Dfyn == 0 or Dgyn == 0:
            print('Zero derivative. No solution found.')
            return None
        xyn = xyn - (Jinv*F_mat)
    print('Exceeded maximum iterations. No solution found.')
    return None
print("theta is ",theta)    
print("x and y are",newton(x,y,10**-10,15))