from mpEntropy import *

np.set_printoptions(linewidth=120)

sysVar = mpSystem("sayak.ini")

print("Dimension of the basis:", sysVar.dim)

# initially occupied states with relative weight (entanglement of starting state):
initstates = [[(sysVar.N, 0, 0, 0, 0), 1]]

J = 0.5

for i in range(sysVar.m):
    for j in range(sysVar.m):
        if i == j:
            sysVar.hamiltonian += sysVar.operators[i, i]
        elif abs(i-j) == 1:
            sysVar.hamiltonian += J * sysVar.operators[i, j]

#sysVar.initHamiltonian()
sysVar.initEvolutionMatrix()

# start with all particles in 0th state
for el in initstates:
    tmp = sysVar.basisDict[el[0]]
    sysVar.state[tmp] = el[1]

#sysVar.expectValue(sysVar.operators[0, 0])

sysVar.evolve()

sysVar.plot()
