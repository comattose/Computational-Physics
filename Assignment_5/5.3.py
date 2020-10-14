#HW5.3 Jack James

import xlrd
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

d1 = pd.read_csv(r'C:/Users/Jack/Downloads/data_1.csv')
d2 = pd.read_csv(r'C:/Users/Jack/Downloads/data_2.csv')
d3 = pd.read_csv(r'C:/Users/Jack/Downloads/data_3.csv')


#for i in range(1,4):
#    plt.subplot(2,3,i)
#    plt.text(0.5,0.5,str((i)),fontsize=18,ha='center')
    
#fig, axs = plt.subplots(3)
#axs[0].plot(d1[('x','y')])
#axs[1].plot(d2.plot('x','y'))
#axs[2].plot(d3.plot('x','y'))


# plot dataframes 
ax1 = d1.plot("x", "y")

#plt.show()

ax2 = d2.plot("x", "y")

#plt.show()

ax3 = d3.plot("x", "y")

#plt.show()
#ax1.plot()
#ax2.plot()
#ax3.plot()

