from mpEntropy import *
np.set_printoptions(linewidth=120)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--plot", help="plot only", action="store_true")
parser.add_argument("-pd","--plot-data", help="plot data", action="store_true")
parser.add_argument("-pa","--plot-animation", help="plot animation", action="store_true")
parser.add_argument("-pt","--plot-timescale", help="plot timescale", action="store_true")
args = parser.parse_args()

#coarse benchmarking for preparations
t0 = tm.time()

if args.plot or args.plot_data or args.plot_animation or args.plot_timescale:
    plotBool = True
else:
    plotBool = False

sysVar = mpSystem("config.ini",plotOnly=plotBool)

if args.plot:
    sysVar.plot()
    exit()

if args.plot_data:
    if sysVar.boolPlotData:
        sysVar.plotData()
    exit()

if args.plot_timescale:
    sysVar.plotTimescale()
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
t = 0.1

# quartic prefactors
u_nm = 1e-2
u_nn = 10 * u_nm

for i in range(0, sysVar.m):
    for j in range(0, sysVar.m):
        if i!=j:
            sysVar.hamiltonian += 1 * sysVar.operators[i,j]
        else:
            sysVar.hamiltonian += i * (dE) * sysVar.operators[i,j]

tmp = np.matrix( np.zeros( (sysVar.dim , sysVar.dim) ) )
for i in range(0, sysVar.m):
    for j in range(0, sysVar.m):
        for k in range(0, sysVar.m):
            for l in range(0, sysVar.m):
                tmp = getQuartic(sysVar,i,j,k,l)
                if  i==j and k==l and k==j:
                    sysVar.hamiltonian += (u_nn) * tmp
                else:
                    sysVar.hamiltonian += (u_nm) * tmp
                del tmp

print('The Hamiltonian has been written!')

sysVar.initEvolutionMatrix(3)

#initially occupied states with relative weight (entanglement of starting state):
initstates = [[(sysVar.N,0,0,0,0),1]]

#start with all particles in 0th state
for el in initstates:
    tmp = sysVar.basisDict[el[0]]
    sysVar.state[tmp, 0] = el[1]
sysVar.normalize(True)

#sysVar.stateEnergy(muperc=30,sigma=50,altSign=True,skip=0,dist='std',muoff=[0],peakamps=[1],skew=10)
print("Preparations finished after " + time_elapsed(t0,60,0) + " \n")

sysVar.evolve()
sysVar.plot()
