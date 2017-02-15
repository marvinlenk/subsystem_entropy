from mpEntropy import *


np.set_printoptions(linewidth=120)
   
#start benchmarking
t0 = tm.time()

sysVar = mpSystem("interact_0.ini")
print("Dimension of the basis:",sysVar.dim)

#sysVar.hamiltonian = quartArr[1,0,0,1]
#sysVar.hamiltonian = quadArr[1,0]
#sysVar.initEvolutionMatrix()
#sysVar.plotHamiltonian()
#exit()

#example hamiltonian with interaction

# onsite energy distance
dE = 1
# transition element magnitude, will be lowered by distance
t = 1
u = 1e-1

for i in range(0, sysVar.m):
    for j in range(0, sysVar.m):
        if i!=j:
            sysVar.hamiltonian += (t) * sysVar.operators[i,j]
        else:
            sysVar.hamiltonian += i * dE * sysVar.operators[i,j]

tmp = np.matrix( np.zeros( (sysVar.dim , sysVar.dim) ) )
for i in range(0, sysVar.m):
    for j in range(0, sysVar.m):
        for k in range(0, sysVar.m):
            for l in range(0, sysVar.m):
                tmp = getQuartic(sysVar,i,j,k,l)
                if  i==l and j==k:
                    sysVar.hamiltonian += u * tmp
                else:
                    sysVar.hamiltonian += (u) * tmp
                del tmp

print('The Hamiltonian has been written!')

tfil = open('./eigvals.txt','w')                    
for el in la.eigvalsh(sysVar.hamiltonian.toarray()):
    tfil.write(str(el)+'\n')
tfil.close()

sysVar.initEvolutionMatrix(3)
#initially occupied states with relative weight:
initstates = [[(sysVar.N,0,0,0),1]]

#start with all particles in 0th state
for el in initstates:
    tmp = sysVar.basisDict[el[0]]
    sysVar.state[tmp, 0] = el[1]
sysVar.plotHamiltonian()
sysVar.normalize(True)
print("Preparations finished after " + time_elapsed(t0,1,4) + " \n")

sysVar.evolve()
sysVar.plot()
