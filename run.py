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

dp=10**5






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
T = U(H, 0.05)
S = T.getH()
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
j=1
ni = sysVar.operators[i,i]
nj = sysVar.operators[j,j]
P=32
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
    l1  = []
    l2  = []
    st1 = T.dot(state_1)
    st2 = T.dot(state_2)
    ex1 = vdot(st1, ni.dot(st1))
    ex2 = vdot(psi, nj.dot(psi))
    v1 = vdot(st1, ni.dot(st2))
    v2 =  vdot(st1, ni.dot(st2)) - ex1 * ex2
    l1.append((0.05, v1.real, v2.real))
    l2.append((0.05, v1.imag, v2.imag))
    V =U1
    for i in range(P):
        st1 = V.dot(st1)
        st2 = V.dot(st2)
        ex1 = vdot(st1, ni.dot(st1))
        ex2 = vdot(psi,nj.dot(psi))
        v1 = vdot(st1, ni.dot(st2))
        v2 = vdot(st1, ni.dot(st2)) - ex1 * ex2
        l1.append(((i+1)*0.1+0.05, v1.real, v2.real))
        l2.append(((i + 1) * 0.1 + 0.05, v1.imag, v2.imag))

        V = V.dot(U0)
    return l1, l2
#calculates the negative function for negative tau

def f2(P):
    l1 = []
    l2 = []
    st1 = S.dot(state_1)
    st2 = S.dot(state_2)
    ex1 = vdot(st1, ni.dot(st1))
    ex2 = vdot(psi, nj.dot(psi))
    v1 = vdot(st1, ni.dot(st2))
    v2 = vdot(st1, ni.dot(st2)) - ex1 * ex2
    l1.append((-0.05, v1.real, v2.real))
    l2.append((-0.05, v1.imag, v2.imag))
    W = U2
    for i in range(P):
        st1 = W.dot(st1)
        st2 = W.dot(st2)
        ex1 = vdot(st1, ni.dot(st1))
        ex2 = vdot(psi, nj.dot(psi))
        v1 = vdot(st1, ni.dot(st2))
        v2 = vdot(st1, ni.dot(st2)) - ex1 * ex2
        l1.append((-(i + 1) * 0.1 - 0.05, v1.real, v2.real))
        l2.append((-(i + 1) * 0.1 - 0.05, v1.imag, v2.imag))

        W = W.dot(U2)
    l1 = list(reversed(l1))
    l2 = list(reversed(l2))
    return l1, l2



#print(f2(P)[0])

#print(f1(P)[0])

rl = f2(P)[0]+f1(P)[0]

im = f2(P)[1]+ f1(P)[1]

np.savetxt('real.txt', rl)
np.savetxt('imaginary.txt', im)





#rl= f2(P)[0].extend(f1(P)[0])

#im= f2(P)[1].extend(f1(P)[1])


















#sysVar.expectValue(den_corr_ops(i,j))


#sysVar.evolve()

#sysVar.plot()
