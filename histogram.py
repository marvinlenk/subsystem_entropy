import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from math import*

l=np.loadtxt('energies_N25_Delta10_U1_Uprime0.1.dat')

def func(s,b,A):
    return A*(s**b)*np.exp(-(pi**2/16.0)*b*(s**2)) #-(B-(pi*b/4.0)*s))




def lev_sp_hist(l,N):
    ar = []
    for i in range(max([1000,N-5000]) , min([len(l)-1000, N+5000])):
        ar.append(l[i])

    lsp=[]
    for i in range(len(ar)-1):
        lsp.append(ar[i+1]-ar[i])

    hst = np.histogram(lsp,bins=1000)

    x = hst[1]
    x = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
    y = hst[0]
    N = sum(y)
    y = [y[i]/N for i in range(len(y))]
    mu = sum([x[i]*y[i] for i in range(len(y))])
    x = [x[i]/mu for i in range(len(x))]
    #print (x,y)

    plt.plot(x, y, label='histogram')


    params, error = optimize.curve_fit(func, x, y, p0=[1.0,100.0])

    print(params)

    f = [func(t,params[0], params[1]) for t in x]

    plt.plot(x, f , label='Fitted function')

    plt.savefig('./plots/plot.eps', format='eps', dpi=1000)

    plt.legend()

lev_sp_hist(l,15000)