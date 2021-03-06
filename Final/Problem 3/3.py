#Final - Problem 3 - Jack James
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize.minpack import leastsq
import pandas as pd
#It takes like 10 minutes for me to get a result. I do not know how to optimize this.


#this computes the energy.  Only use this once.  
def get_init_energy(s,J,b,N,PERIODICBOUNDARIES):
	e=0
	for i in range(N-1):
		for j in range(N-1):
			e+=-s[i][j]*s[i+1][j]
			e+=-s[i][j]*s[i][j+1]
	# bulk contribution
	for i in range(N-1):
		e+=-s[i][N-1]*s[i+1][N-1]
		if(PERIODICBOUNDARIES):
			e+=-s[i][N-1]*s[i][0]
	#right side
	
	for j in range(N-1):
		e+=-s[N-1][j]*s[N-1][j+1]
		if(PERIODICBOUNDARIES):
			e+=-s[N-1][j]*s[0][j]
	#bottom
	
	if(PERIODICBOUNDARIES):
		e+=-s[N-1][N-1]*s[N-1][0]
		e+=-s[N-1][N-1]*s[0][N-1]
	return J*e	
	
#this computes the energy difference due to a spin flip
def energy_difference(s1,s2,spins,J,b,N,PERIODICBOUNDARIES):
	de=0

	evaluate=True
	t1=s1+1
	t2=s2
	if(t1==N):
		if(PERIODICBOUNDARIES):
			t1=0
		else:
			evaluate=False
			
	if evaluate:
		de+=2*spins[s1][s2]*spins[t1][t2]
		
	
	
	evaluate=True
	t1=s1-1
	t2=s2
	if(t1==-1):
		if(PERIODICBOUNDARIES):
			t1=N-1
		else:
			evaluate=False
			
	if evaluate:
		de+=2*spins[s1][s2]*spins[t1][t2]
		
		
	evaluate=True
	t1=s1
	t2=s2+1
	if(t2==N):
		if(PERIODICBOUNDARIES):
			t2=0
		else:
			evaluate=False
			
	if evaluate:
		de+=2*spins[s1][s2]*spins[t1][t2]

	
	
	evaluate=True
	t1=s1
	t2=s2-1
	if(t2==-1):
		if(PERIODICBOUNDARIES):
			t2=N-1
		else:
			evaluate=False
			
	if evaluate:
		de+=2*spins[s1][s2]*spins[t1][t2]
		
	de=de*J;
	
	if spins[s1][s2]>0:
		de+=2*b
	else:
		de-=2*b
	
	return de

#this flips the spin and checks for the change in all of its neighbors.

#metropolis criterion
def metropolis(e,enew):  
	if(e>enew):
		return True
	else:
		if np.random.random()<math.exp((e-enew)):
			return True
		else:
			return False


N=50  #size of the lattice
nstep=5*10**5	#steps in the simulation
printstep=2000	#steps between printing
nprint=int(nstep/printstep)
nrun=100

t=np.zeros(nprint)
avm=np.zeros(nprint)
ave=np.zeros(nprint)


np.random.seed(2)

J=0.1  #1st interaction strength (with kT=1)
b=0  #magnetic field.  This is h in class.  

pbc=True  #this flags whether to use PBCs

for run in range(nrun):

		
	spins=[[0 for _ in range(N)] for _ in range(N)]  #define the spins as all down
	m=0  #initial magnetization is all down
	for i in range(N):
		for j in range(N):
			if np.random.random()<.5:
				spins[i][j]=1  
				m+=1
			else:
				spins[i][j]=-1
				m-=1;			
	e=get_init_energy(spins,J,b,N,pbc)
	#compute initial energy


	for step in range(nstep):  #iterate over steps
		s1=np.random.randint(0,N)
		s2=np.random.randint(0,N)
		#randomly choose a spin to flip.  		
		
		
		enew=e+energy_difference(s1,s2,spins,J,b,N,pbc)
		#compute the energy
		if(metropolis(e,enew)):
		#with the metropolis criterion, choose to flip the spin
			spins[s1][s2]=-spins[s1][s2]
			if spins[s1][s2]>0:
				m+=2
			else:
				m-=2
			#update the magnetization due to the flip
			e=enew
			#update the energy due to the flip
	
		if step%printstep==0:
			t[int(step/printstep)]=step
			#every printstep, we want to print data
			avm[int(step/printstep)]+=abs(m)/N/N
			ave[int(step/printstep)]+=e/J/N/N

s=np.transpose(spins)
for i in range(N):
	y=[j for j,e in enumerate(s[i]) if e==1]
	x=[i for _ in range(len(y))]
	plt.scatter(x,y,color='b')
ax1=plt.axes()
ax1.set_aspect('equal')
plt.show()


plt.plot(t/nrun,avm/nrun)
plt.show()
plt.plot(t/nrun,ave/nrun)
plt.show()

t=np.zeros(nprint)
avm=np.zeros(nprint)
ave=np.zeros(nprint)


np.random.seed(2)

J=0.2  #2nd interaction strength (with kT=1)
b=0  #magnetic field.  This is h in class.  

pbc=True  #this flags whether to use PBCs

for run in range(nrun):

		
	spins=[[0 for _ in range(N)] for _ in range(N)]  #define the spins as all down
	m=0  #initial magnetization is all down
	for i in range(N):
		for j in range(N):
			if np.random.random()<.5:
				spins[i][j]=1  
				m+=1
			else:
				spins[i][j]=-1
				m-=1;			
	e=get_init_energy(spins,J,b,N,pbc)
	#compute initial energy


	for step in range(nstep):  #iterate over steps
		s1=np.random.randint(0,N)
		s2=np.random.randint(0,N)
		#randomly choose a spin to flip.  		
		
		
		enew=e+energy_difference(s1,s2,spins,J,b,N,pbc)
		#compute the energy
		if(metropolis(e,enew)):
		#with the metropolis criterion, choose to flip the spin
			spins[s1][s2]=-spins[s1][s2]
			if spins[s1][s2]>0:
				m+=2
			else:
				m-=2
			#update the magnetization due to the flip
			e=enew
			#update the energy due to the flip
	
		if step%printstep==0:
			t[int(step/printstep)]=step
			#every printstep, we want to print data
			avm[int(step/printstep)]+=abs(m)/N/N
			ave[int(step/printstep)]+=e/J/N/N

s=np.transpose(spins)
for i in range(N):
	y=[j for j,e in enumerate(s[i]) if e==1]
	x=[i for _ in range(len(y))]
	plt.scatter(x,y,color='b')
ax1=plt.axes()
ax1.set_aspect('equal')
plt.show()


plt.plot(t/nrun,avm/nrun)
plt.show()
plt.plot(t/nrun,ave/nrun)
plt.show()

t=np.zeros(nprint)
avm=np.zeros(nprint)
ave=np.zeros(nprint)


np.random.seed(2)

J=0.3  #3rd interaction strength (with kT=1)
b=0  #magnetic field.  This is h in class.  

pbc=True  #this flags whether to use PBCs

for run in range(nrun):

		
	spins=[[0 for _ in range(N)] for _ in range(N)]  #define the spins as all down
	m=0  #initial magnetization is all down
	for i in range(N):
		for j in range(N):
			if np.random.random()<.5:
				spins[i][j]=1  
				m+=1
			else:
				spins[i][j]=-1
				m-=1;			
	e=get_init_energy(spins,J,b,N,pbc)
	#compute initial energy


	for step in range(nstep):  #iterate over steps
		s1=np.random.randint(0,N)
		s2=np.random.randint(0,N)
		#randomly choose a spin to flip.  		
		
		
		enew=e+energy_difference(s1,s2,spins,J,b,N,pbc)
		#compute the energy
		if(metropolis(e,enew)):
		#with the metropolis criterion, choose to flip the spin
			spins[s1][s2]=-spins[s1][s2]
			if spins[s1][s2]>0:
				m+=2
			else:
				m-=2
			#update the magnetization due to the flip
			e=enew
			#update the energy due to the flip
	
		if step%printstep==0:
			t[int(step/printstep)]=step
			#every printstep, we want to print data
			avm[int(step/printstep)]+=abs(m)/N/N
			ave[int(step/printstep)]+=e/J/N/N

s=np.transpose(spins)
for i in range(N):
	y=[j for j,e in enumerate(s[i]) if e==1]
	x=[i for _ in range(len(y))]
	plt.scatter(x,y,color='b')
ax1=plt.axes()
ax1.set_aspect('equal')
plt.show()


plt.plot(t/nrun,avm/nrun)
plt.show()
plt.plot(t/nrun,ave/nrun)
plt.show()

t=np.zeros(nprint)
avm=np.zeros(nprint)
ave=np.zeros(nprint)


np.random.seed(2)

J=0.4  #4th interaction strength (with kT=1)
b=0  #magnetic field.  This is h in class.  

pbc=True  #this flags whether to use PBCs

for run in range(nrun):

		
	spins=[[0 for _ in range(N)] for _ in range(N)]  #define the spins as all down
	m=0  #initial magnetization is all down
	for i in range(N):
		for j in range(N):
			if np.random.random()<.5:
				spins[i][j]=1  
				m+=1
			else:
				spins[i][j]=-1
				m-=1;			
	e=get_init_energy(spins,J,b,N,pbc)
	#compute initial energy


	for step in range(nstep):  #iterate over steps
		s1=np.random.randint(0,N)
		s2=np.random.randint(0,N)
		#randomly choose a spin to flip.  		
		
		
		enew=e+energy_difference(s1,s2,spins,J,b,N,pbc)
		#compute the energy
		if(metropolis(e,enew)):
		#with the metropolis criterion, choose to flip the spin
			spins[s1][s2]=-spins[s1][s2]
			if spins[s1][s2]>0:
				m+=2
			else:
				m-=2
			#update the magnetization due to the flip
			e=enew
			#update the energy due to the flip
	
		if step%printstep==0:
			t[int(step/printstep)]=step
			#every printstep, we want to print data
			avm[int(step/printstep)]+=abs(m)/N/N
			ave[int(step/printstep)]+=e/J/N/N

s=np.transpose(spins)
for i in range(N):
	y=[j for j,e in enumerate(s[i]) if e==1]
	x=[i for _ in range(len(y))]
	plt.scatter(x,y,color='b')
ax1=plt.axes()
ax1.set_aspect('equal')
plt.show()


plt.plot(t/nrun,avm/nrun)
plt.show()
plt.plot(t/nrun,ave/nrun)
plt.show()

print('a) The fluctuations definitely do significantly depend on J. The whole plot jumps drastically as J increases.')

#sef = np.polyfit(t/nrun,ave/nrun,1) #data for single exponential fit: Ae**(-n/tauc)+B
#sef_function = np.exp(sef[1])*np.exp(sef[0]*t/nrun)
#plt.plot(t/nrun, ave/nrun, "o")
#plt.plot(t/nrun, sef_function)
print('b) They vary with J and do not seem to fit very well.')

#def1 = np.polyfit(t/nrun,ave/nrun,2) #data for double exponential fit: Aexp(-n/m1)+Bexp(-n/m2). The 1 is just there because def is define.
#def_function = np.exp(def1[1])*np.exp(def1[0]*t/nrun)
#plt.plot(t/nrun, ave/nrun, "o")
#plt.plot(t/nrun, def_function)
#Going to need to come back to this. Not sure how to adapt for two exponentials.

## regression functions
def single_exp(A, B, Tauc,):
    """
    Exponential function y = A * exp(n / Tauc) + B
    """
    return lambda n: A * np.exp(n / Tauc) + B

def double_exp(A, B, m1, m2):
    """
    Exponential function y = A * exp(n / m1) + B * exp(n / m2)
    """
    return lambda n: A * np.exp(n / m1) + B * np.exp(n / m2)


## interpolation
def interpolate(x, df, fun):
    """
    Interpolate Y from X based on df, a dataframe with columns 'x' and 'y'.
    """
    resid = lambda p, x, y: y - fun(*p)(x)
    ls = leastsq(resid, [1.0, 1.0, 1.0, 1.0], args=(df['x'], df['y']))
    a, b, c, d = ls[0]
    y = fun(a, b, c, d)(x)
    return y

data = {'x': [t/nrun],'y': [ave/nrun]}               #creating pandas dataframe with my result so I can interpolate
df = pd.DataFrame (data, columns = ['x','y'])

plt.plot(t/nrun, ave/nrun, "o")
plt.plot(t/nrun, interpolate(ave/nrun,df,single_exp))
plt.plot(t/nrun, interpolate(ave/nrun,df,double_exp))
print('c) Fit is slightly worse? ')

print('d) They are similar.')
