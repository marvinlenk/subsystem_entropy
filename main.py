from mpEntropy import *

np.set_printoptions(linewidth=120)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plot", help="plot only", action="store_true")
parser.add_argument("-pd", "--plot-data", help="plot data", action="store_true")
parser.add_argument("-pt", "--plot-timescale", help="plot timescales", action="store_true")
parser.add_argument("-pa", "--plot-animation", help="plot animation", action="store_true")
parser.add_argument("-poe", "--plot-occen", help="plot occupation operator matrices in energy eigenbasis",
                    action="store_true")
parser.add_argument("-pods", "--plot-odsingles", help="plot single off diagonals", action="store_true")
args = parser.parse_args()

# coarse benchmarking for preparations
t0 = tm.time()

if args.plot or args.plot_data or args.plot_animation or args.plot_occen or args.plot_odsingles or args.plot_timescale:
    plotBool = True
else:
    plotBool = False

sysVar = mpSystem("default.ini", plotOnly=plotBool)

if args.plot:
    sysVar.plot()
    exit()

if args.plot_data:
    if sysVar.boolPlotData:
        sysVar.plotData()
    exit()

if args.plot_timescale:
    if sysVar.boolPlotTimescale:
        sysVar.plotTimescale()
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
    if sysVar.boolPlotOffDiagOccSingles:
        sysVar.plotOffDiagOccSingles()
    exit()

print("Dimension of the basis:", sysVar.dim)

# initially occupied states with relative weight (entanglement of starting state):
initstates = [[(sysVar.N, 0, 0, 0), 1]]

sysVar.initAllHamiltonians()
sysVar.initAllEvolutionMatrices()

# start with all particles in 0th state
for el in initstates:
    tmp = sysVar.basisDict[el[0]]
    sysVar.state[tmp] = el[1]
sysVar.normalize(True)

sysVar.evolve()

if sysVar.boolRetgreen:
    sysVar.evaluateGreen()

sysVar.plot()
