import numpy as np
import matplotlib.pyplot as plt
from random import random
l=[]


N=100

def lev_sp_hist(l):
    ar = []
    for i in range(101,len(l)-100):
        ar.append(l[i])

    lsp=[]
    for i in range(len(ar)-1):
        lsp.append(ar[i+1]-ar[i])

    hst = np.histogram(lsp,bins=1000)

    x = hst[1]
    x = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
    y = hst[0]

    print (x,y)

    plt.plot(x, y, label='histogram')

    plt.savefig('./plots/plot.eps', format='eps', dpi=1000)


lev_sp_hist(l)