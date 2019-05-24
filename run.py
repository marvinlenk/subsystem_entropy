from mpEntropy import *
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

np.set_printoptions(linewidth=120)

sysVar = mpSystem("sayak.ini")

print("Dimension of the basis:", sysVar.dim)


sysVar.initHamiltonian()
H=csc_matrix(sysVar.hamiltonian)

dp=0






#sysVar.initEvolutionMatrix()



#initial state
psi0=np.zeros(sysVar.dim)
psi0[0]=1.0
print(psi0)


#evolution operator
def U(H,t):
    A = np.multiply(-1j*t, H)
    return linalg.expm(A)

#density density corr ops
def den_corr_ops(i, j):
    return sysVar.operators[i, i].dot(sysVar.operators[j, j])



psi=psi0

#l0=range(dp)

#l1 = np.zeros(dp)
#l2 = np.zeros(dp)
#l3 = np.zeros(dp)
#l4 = np.zeros(dp)

U0 = U(H, 0.1)

#evolution routine


#for k in range(dp):
#    psi=U0.dot(psi)
#    nrm = la.norm(psi)
#    print(nrm)
#    dnm = nrm**2
#    n0 = vdot(psi, sysVar.operators[0,0].dot(psi))/dnm
#    n1 = vdot(psi, sysVar.operators[1,1].dot(psi))/dnm
#    n2 = vdot(psi, sysVar.operators[2,2].dot(psi))/dnm
#    n3 = vdot(psi, sysVar.operators[3,3].dot(psi))/dnm
#    n4 = vdot(psi, sysVar.operators[4,4].dot(psi))/dnm
#    l1[k] = ((vdot(psi, den_corr_ops(0, 1).dot(psi))) / dnm) - n0*n1
#    l2[k] = ((vdot(psi, den_corr_ops(0, 2).dot(psi))) / dnm) - n0*n2
#    l3[k] = ((vdot(psi, den_corr_ops(0, 3).dot(psi))) / dnm) - n0*n3
#    l4[k] = ((vdot(psi, den_corr_ops(0, 4).dot(psi))) / dnm) - n0*n4


#plt.plot(l0 , l1, label='corr(0,1)')
#plt.plot(l0 , l2, label='corr(0,2)')
#plt.plot(l0 , l3, label='corr(0,3)')
#plt.plot(l0 , l4, label='corr(0,1)')

#plt.xlabel('time')
#plt.ylabel('correlations')
#plt.legend()


#plt.savefig('./plots/plot.eps', format='eps', dpi=1000)


#two time correlation function routine
i=0
j=0
ni = sysVar.operators[i,i]
nj = sysVar.operators[j,j]
P=20
#evolve the initial state to t=t

U0 = U(H, 0.1)
for k in range(dp):
    psi = U0.dot(psi)

nrm = la.norm(psi)
psi = np.multiply(1.0/nrm, psi)

state_1 = psi
state_2 = nj.dot(psi)

U1=U0

U2=U0.getH()

#calculates the correlation function for positive tau
def f1(P):
    V =U1
    for i in range(P):
        st1 = V.dot(state_1)
        st2 = V.dot(state_2)

        print ( 0.1*(2**i),  vdot(st1, ni.dot(st2)))
        V = V.dot(V)

#calculates the negative function for negative tau

def f2(P):
    W = U2
    for i in range(P):
        st1 = W.dot(state_1)
        st2 = W.dot(state_2)

        print ( -0.1*(2**i),  vdot(st1,ni.dot(st2)))
        W = W.dot(W)

f1(P)
f2(P)
















#sysVar.expectValue(den_corr_ops(i,j))


#sysVar.evolve()

#sysVar.plot()
