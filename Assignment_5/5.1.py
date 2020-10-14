#HW5.1 Jack James

import numpy as np

print("This takes a really long time to compile (like an hour), should I lower my root windows since know what roots are?")
print("(Since this would be used mainly when we don't know what roots are.)")
print("I went ahead and lowered windows to one to try and speed it up.")


def func(a):                      #setting up function to make if statements smaller
    return np.exp(a) - a**4



a = -1.5  #first region set for root in vicinity of -1
b = -0.5

if func(a)*func(b) < 0: 
    while abs(b-a) > 10**(-5):
        m = (a+b)/2       #midpoint
        if func(a)*func(b) < 0:
            b = m         #sets b as previous midpoint if not after b
        elif func(b)*func(m) < 0:
            a = m         #sets a as previous midpoint if not before a
    k = (a+b)/2
    print(k)
else:
    print("Not able to find root")
 

    
a = 8.5    #second region set for root in vicinity of 9
b = 9.5

if func(a)*func(b) < 0: 
    while abs(b-a) > 10**(-5):
        m = (a+b)/2         #midpoint
        if func(a)*func(b) < 0:
            b = m
        elif func(b)*func(m) < 0:
            a = m
    k = (a+b)/2
    print(k)
else:
    print("Not able to find root")


    
a = 0.5  #third region set for root in vicinity of 1
b = 1.5

if func(a)*func(b) < 0: 
    while abs(b-a) > 10**(-5):
        m = (a+b)/2             #midpoint
        if func(a)*func(b) < 0:
            b = m               
        elif func(b)*func(m) < 0:
            a = m
    k = (a+b)/2
    print(k)
else:
    print("Not able to find root")

