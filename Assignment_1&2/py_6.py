import numpy as np
import math
import matplotlib.pyplot as plt

#  FUNCTION DEFINITION
def logistic_calculate(r,x):
	#r and x are both passed by value, and cannot be updated within the function
	return r*x*(1-x)
	#this returns the new value of x.
# END OF FUNCTION DEFINITION


# PARAMETERS
#max_iteration=10000
max_iteration=50  # how many iterations to do
max_period=256  # largest period
start_testing=max_iteration-max_period	 #how many numbers to print to the screen
numdiv=5000
tolerance=1e-8


bfile=open('bifurcations_python.txt','w')  #where to write bifurcation data
pfile=open('periods_python.txt','w')		#where to write period data


a=np.zeros(50)
b=np.zeros(50)
c=np.zeros(50)
d=np.zeros(50)
	
for riter in range(numdiv):
    r_1=2
	
	# INITIALIZATION
    x=0.01    # initial value for x
    xobs=0
    period=max_period


    for iter in range(max_iteration):
        x=logistic_calculate(r_1,x)
		
        if iter==start_testing:
            xobs=x
            bfile.write(str(r_1))
		
        if iter>start_testing:
            bfile.write(","+str(x))
            if (abs(x-xobs)<tolerance) and (period==max_period):
                period=iter-start_testing
		
    pfile.write(str(r_1)+","+str(period));
    a[iter]=period
    b[iter]=x

plt.plot(b,a)

for riter in range(numdiv):
    r_2=2.99
	
	# INITIALIZATION
    x=0.01    # initial value for x
    xobs=0
    period=max_period


    for iter in range(max_iteration):
        x=logistic_calculate(r_2,x)
		
        if iter==start_testing:
            xobs=x
            bfile.write(str(r_2))
            
		
        if iter>start_testing:
            bfile.write(","+str(x))
            if (abs(x-xobs)<tolerance) and (period==max_period):
                period=iter-start_testing
		
    pfile.write(str(r_2)+","+str(period));
    c[iter]=period
    d[iter]=x

plt.plot(d,c)

print('\n',a,'\n',b,'\n',c,'\n',d)

    
bfile.close();
pfile.close();
			
		
