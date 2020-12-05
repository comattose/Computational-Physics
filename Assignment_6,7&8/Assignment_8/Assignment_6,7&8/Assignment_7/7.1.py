import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import cm

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

#Derivative functions
def derivative(X,t,):
	return np.array([X[1],-np.sin(X[0])])
def derivative2(A,B,C):
    return np.array(d2rdt2(A,B,C))


# Parameters
N = 1000
L = 1.                # BC of -L/2 to L/2
h = L/(N-1)            
tau = .01  #time step

xs = np.arange(N)*h - L/2    #x terms


times = np.zeros(N)          #t terms
for i in range(N):
    times[i] = times[i-1] + tau

Tp, Xp = np.meshgrid(times, xs)


M = [1,10,100]
for D in M:
    
    
    rho = np.zeros(N)
    rho[0] = 1

    
    A = np.zeros((N,N))     # Reset matrix
    coeff = D/(2*h)
    for i in range(1,N-1) :
        A[i,i-1] = coeff
        A[i,i] = -2*coeff   
        A[i,i+1] = coeff

    
    A[0,-1] = coeff;   A[0,0] = -2*coeff;     A[0,1] = coeff
    A[-1,-2] = coeff;  A[-1,-1] = -2*coeff;   A[-1,0] = coeff

    # Crank-Nicolson that
    dCN = np.dot( np.linalg.inv(np.identity(N) - .5*tau*A - .5*tau*np.identity(N)), 
                (np.identity(N) + .5*tau*A + .5*tau*np.identity(N)) )

    rhos = []
    for i in range(N):
        rho = np.dot(dCN,rho)
        rhos.append(rho)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.contour3D(Tp, Xp, rhos, 50, cmap=cm.cool)
    ax.set_xlabel("T")
    ax.set_ylabel("X")
    ax.set_zlabel("Rho")
    ax.set_title('Diffusion')
    ax.text(1,1,22,D)
    ax.legend()
    plt.show()



