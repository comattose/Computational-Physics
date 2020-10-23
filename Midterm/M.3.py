#Midterm Problem #3 - Jack James


import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import csv
import sys
from sympy import symbols, diff 
from scipy.fft import fft, ifft


#PART A (new) 
    #I tried so many ways of importing this and kept getting various issues at later calculations. When I finally got it working, I had 
#so many commented-out lines that I ended up forgetting what blocks I needed and what was experimentation. I have tried my best to reformat 
#acting code, but I may have left explanation in below comments.

DL = []

with open('data.csv','r') as data:         #another way to import code without transposing
    df = csv.reader(data)
    for i in df:
        DL.extend(i)
#print("this is DL",DL)
array = np.array(DL)
arr = array.astype(np.float)
N = arr.size#()
Fourier = np.fft.fft(arr)

Za = abs(Fourier/N)
midpoint = int(N/2)
Z = Za[:midpoint]
Z[1:] = 2*Z[1:]
W = 1
F = np.arange(N/2)
#print("this is F",F)
for i in range(int(N/2)):
    F[i] = F[i]/N
#print("F is",F)

#plt.figure("FFT")
#plt.plot(F,Z)

plt.show()

#PART B (new)
Y = np.zeros(N)
#print(Y)
#print(DL,type(DL))

for i in range(N):
    if i < N-1:
        y = 0.5*(float(DL[i])+float(DL[i+1]))
        Y[i] = y
    elif i == N-1:
        Y[i] = Y[0]
#print(Y)

Fourier2 = np.fft.fft(Y)
Z2a = abs(Fourier2/N)
midpoint2 = int(N/2)
Z2 = Z2a[:midpoint2]
Z2[1:] = 2*Z2[1:]

F2 = np.arange(N/2)
for i in range(int(N/2)):
    F2[i] = F2[i]/N
#print("this is F2",F2)

#plt.figure("FFT2")
#plt.plot(F2,Z2)

print("This is an example of a low-pass filter because as frequency increases, output goes to zero.")

#Dual Plots

fig, axs = plt.subplots(2)                                        #So when I originally tried to plot without x2/2, both were identical.
#fig.subtitle('Raw Data vs Fourier Transform')
axs[0].plot(F,Z)
axs[1].plot(F2,Z2)

#PART A (old, older, ancient)
#data = pd.read_csv (r'C:\Users\Jack\Documents\Fall 2020\Comp Phys\Assignments\Midterm\data_transposed.csv')   #importing data
#df = pd.DataFrame(data, columns= ['x'])       #choosing only first column, had to edit excel file by adding header. Comments at bottom were
#print ("this is df",df)                       #previous attempt which gave really weird inputs. Insisted on taking empty column values as nan.
#df.plot()                                     #now this is causing issues with part B. Need to find better import > array/list method.

#fourier = fft(df)                             #gives same result but complex?
#fft_2 = fft(df[:2])                           #this attempt just deleted negative amplitude portion
#fft_3 = np.abs(np.fft.fft(df))*2
#freq_3 = np.fft.fftfreq(df.size)            #not using time step since fourier of data
#idx = np.argsort(freq_3)
#plt.plot(abs(fft_2))
#plt.show()
#print("this is fourier", fourier)            

#plt.plot(df)
#plt.plot(fourier)

#real_fft = 2.0/(len(df)) * np.abs(fourier[:len(df)//2])  #multiplying x2 and dividing /2 to try to fix odd/even array for transform

#fig, axs = plt.subplots(2)                                        #So when I originally tried to plot without x2/2, both were identical.
#fig.subtitle('Raw Data vs Fourier Transform')
#axs[0].plot(freq_3[idx],df)
#axs[1].plot(fourier)
#axs[1].plot(real_fft)
#axs[1].plot(freq_3[idx],fft_3[idx])

#df = pd.read_csv(r'C:\Users\Jack\Documents\Fall 2020\Comp Phys\Assignments\Midterm\data.csv')
#df = pd.read_csv(r'C:\Users\Jack\Documents\Fall 2020\Comp Phys\Assignments\Midterm\data_transposed.csv')

#print("generic print",df)
#print("shape",df.shape)
#print("tail",df.tail())

#base_data = df.plot('x')

#csv_file = np.genfromtxt(r'C:\Users\Jack\Documents\Fall 2020\Comp Phys\Assignments\Midterm\data_transposed.csv')
#data=csv_file[:].tolist()
#print("Raw data is\n",data)

#base_array=np.asarray([0])

#for i in range(int(127)):
#    base_array=np.append(base_array,data[i])
    
#print("Array of original is:\n", base_array)

#z = float(base_array)
#print('z is\n',z)

#fou = np.fft(data)
#four = np.fft(base_array)
#fourier = np.fft(z)




#PART B
data = []
#with open()

#N = 50
#X = df.values.tolist()
#Y = np.zeros(N)
#print("X",X)
#print(type(X))
#print(Y)
#for i in range(N):
#    if i < N-1:
#        a = df[i]
#        b = df[i+1]
#        y = (float(a) + float(b))/2
#    elif i == N-1:
#        Y[i] = Y[0]

#fft_b = np.abs(np.fft.fft(Y))*2

#print(Y)


#print("sum is",X[0]+X[1])

#def X_sum(a,b):                                         #creating function to calculate y at each step
#    s = X[a] + X[b] 
#    d = s/2
#    return(d)

#Y = [[np.array(X_sum(0,1))]]
#Y = [0]

#for i in range(0, len(X)): 
#    Y[i] = int(df[i]) 



#for i in range(2,df.size):
    #Y[i] = [(X[i]+X[i+1])/2]
    #Y = np.append(Y,[(X[i]+X[i+1])/2])
    #Y = np.append(Y,[X_sum(X[i],X[i+1])])
#    Y[i-1] = [X_sum(X[i],X[i-1])]
#print(Y)





