from mpEntropy import *
np.set_printoptions(linewidth=120)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--plot", help="plot only", action="store_true")
parser.add_argument("-pd","--plot-data", help="plot data", action="store_true")
parser.add_argument("-pa","--plot-animation", help="plot animation", action="store_true")
args = parser.parse_args()

#coarse benchmarking for preparations
t0 = tm.time()

sysVar = mpSystem("config.ini")

if args.plot:
    sysVar.plot()
    exit()

if args.plot_data:
    if sysVar.boolPlotData:
        sysVar.plotData()
    exit()

if args.plot_animation:
    if sysVar.boolPlotDMAnimation:
        sysVar.plotDMAnimation()
    if sysVar.boolPlotDMRedAnimation:
        sysVar.plotDMRedAnimation()
    exit()
    
print("Dimension of the basis:",sysVar.dim)

#example hamiltonian with interaction

# onsite energy distance
dE = 1

# quadratic pre-factors
t = 0.5

# quartic prefactors
u = 1e-2

for i in range(0, sysVar.m):
    for j in range(0, sysVar.m):
        if i!=j:
            sysVar.hamiltonian += t * sysVar.operators[i,j]
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
                    sysVar.hamiltonian += u * tmp
                del tmp

print('The Hamiltonian has been written!')

sysVar.initEvolutionMatrix(3)
'''
#initially occupied states with relative weight (entanglement of starting state):
initstates = [[(sysVar.N,0,0,0),1]]

#start with all particles in 0th state
for el in initstates:
    tmp = sysVar.basisDict[el[0]]
    sysVar.state[tmp, 0] = el[1]
sysVar.normalize(True)
'''
sysVar.stateEnergy(muperc=90,sigma=0.6)
print("Preparations finished after " + time_elapsed(t0,1,4) + " \n")

sysVar.evolve()
sysVar.plot()
