import numpy as np
from numpy import array

a=array([np.random.randint(-9,11),np.random.randint(-9,11),np.random.randint(-9,11)])  #create two vectors with random int
b=array([np.random.randint(-9,11),np.random.randint(-9,11),np.random.randint(-9,11)])  #    between -10 and 10

print(a,b)

v_1=a
seg=(b-a*((a@b)/(a@a)))
c=b@seg[::-1]
v_2=seg*c

print("Two arbitrary vectors are ",v_1," and ", v_2)
print("and their dot product is ",v_1@v_2)