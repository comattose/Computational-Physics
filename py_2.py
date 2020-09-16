x=1
n=1000000  #couldn't compute with this n after 5 minutes of waiting

for i in range(n):
    x=x*2
    
print(x)

import sys  #found this online but it's so small... Could it be the number of digits for int?
y=sys.maxsize
print(y)

x=1.0
n=1015  #significantly smaller n before getting infinite "error"

for i in range(n):
    x=x*2
    
print(x)

#I'm doing all the python first so can't compare to Matlab yet.