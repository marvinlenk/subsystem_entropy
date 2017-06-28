from mpEntropy import *
np.set_printoptions(linewidth=120)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--plot", help="plot only", action="store_true")
parser.add_argument("-pd","--plot-data", help="plot data", action="store_true")
parser.add_argument("-pa","--plot-animation", help="plot animation", action="store_true")
parser.add_argument("-poe","--plot-occen", help="plot occupation operator matrices in energy eigenbasis", action="store_true")
parser.add_argument("-pods","--plot-odsingles", help="plot single off diagonals", action="store_true")
args = parser.parse_args()

#coarse benchmarking for preparations
t0 = tm.time()

if args.plot or args.plot_data or args.plot_animation or args.plot_occen or args.plot_odsingles:
    plotBool = True
else:
    plotBool = False

sysVar = mpSystem("default.ini",plotOnly=plotBool)

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
    
if args.plot_occen:
    if sysVar.boolPlotOccEn:
        sysVar.plotOccEnbasis()
    exit()

if args.plot_odsingles:
    if sysVar.boolPlotOffDiagSingles:
        sysVar.plotOffDiagSingles()
    exit()

print("Dimension of the basis:",sysVar.dim)

#initially occupied states with relative weight (entanglement of starting state):
initstates = [[(sysVar.N,0,0,0,0),1]]

sysVar.initHamiltonian()
if sysVar.boolRetgreen:
    sysVar.initSpecLoHamiltonian()
    sysVar.initSpecHiHamiltonian()
print('The Hamiltonian has been written!')

sysVar.initEvolutionMatrix()
if sysVar.boolRetgreen:
    sysVar.initSpecLoEvolutionMatrix()
    sysVar.initSpecHiEvolutionMatrix()

#start with all particles in 0th state
for el in initstates:
    tmp = sysVar.basisDict[el[0]]
    sysVar.state[tmp, 0] = el[1]
#sysVar.state[0,0] = 1
sysVar.normalize(True)


#sysVar.stateEnergy(muperc=[50,20,25],sigma=[100,0.5,0.5],phase=['rnd','none','none'],skip=[0,0,0],dist=['rnd','std','std'],peakamps=[0.1,1,0.8],skew=[0,0,0])
print("Preparations finished after " + time_elapsed(t0,60,0) + " \n")

sysVar.evolve()
t0 = tm.time()
if sysVar.boolRetgreen:
    sysVar.evaluateGreen()
    print('green: '+time_elapsed(t0,60,0))
sysVar.plot()
