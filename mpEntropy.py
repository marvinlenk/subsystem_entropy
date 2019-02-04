# This program is explicitly written for PYTHON3.X and will not work under Python2.X

import configparser
import os
import time as tm
import numpy as np
import scipy.linalg as la
from numpy import log as nplog
from numpy import sqrt
from numpy.linalg import matrix_power as npmatrix_power
from numpy.linalg import multi_dot
from numpy import vdot
from scipy.sparse import coo_matrix, csr_matrix, block_diag, save_npz, issparse
from scipy.sparse import identity as spidentity
from scipy.special import binom
from scipy.special import erf
from scipy.special.basic import factorial
from multiprocessing import Pool
from multiprocessing import RawArray as mpRawArray

import entPlot as ep


# manyparticle system class
# noinspection PyTypeChecker
class mpSystem:
    # N = total particle number, m = number of states in total, redStates = array of state indices to be traced out
    def __init__(self, cFile="default.ini", dtType=np.complex128, plotOnly=False, greenOnly=False, cores=False):
        t0 = tm.time()
        self.confFile = cFile
        self.greenOnly = greenOnly

        # number of cores might get changed to one later
        if cores:
            self.cores = cores
        else:
            mkl_nt = os.environ.get('MKL_NUM_THREADS')
            omp_nt = os.environ.get('OMP_NUM_THREADS')
            nexp_nt = os.environ.get('NUMEXPR_NUM_THREADS')
            # it will first look for mkl if available
            if mkl_nt is None and mkl_nt is None and nexp_nt is None:
                self.cores = os.cpu_count()
            elif mkl_nt is not None:
                self.cores = int(mkl_nt)
            elif omp_nt is not None:
                self.cores = int(omp_nt)
            else:
                self.cores = int(nexp_nt)

        # here comes a ridiculously long list of variables which are initialized before loading the config
        self.N = 0
        self.m = 0
        self.kRed = None
        # ## hamiltonian parameters
        self.onsite = 0.0
        self.hybrid = 0.0
        self.interequal = 0.0
        self.interdiff = 0.0
        # ## iteration parameters
        self.steps = 0
        self.deltaT = 0.0
        self.order = 0
        self.loOrder = 0
        self.hiOrder = 0
        self.deltaTgreen = 0.0
        self.deltaTaugreen = 0.0
        self.greensteps = 0
        # ## file management
        self.dataPoints = 0
        self.dmFilesSkipFactor = 0
        self.boolClear = False
        self.boolCleanFiles = False
        self.boolDataStore = False
        self.boolDMStore = False
        self.boolDMRedStore = False
        self.boolDMRedDiagStore = False
        self.boolHamilStore = False
        self.boolOccEnStore = False
        self.boolOccEnDiag = False
        self.occEnSingle = 0
        self.boolOffDiagOcc = False
        self.boolOffDiagDens = False
        self.boolOffDiagDensRed = False
        self.boolEngyStore = False
        self.boolDecompStore = False
        self.boolDiagExpStore = False
        self.boolRetgreen = False
        self.dataFolder = ''
        # ## calculation-parameters
        self.boolOnlyRed = False
        self.boolTotalEntropy = False
        self.boolReducedEntropy = False
        self.boolTotalEnergy = False
        self.boolReducedEnergy = False
        self.boolOccupations = False
        self.boolEntanglementSpectrum = False
        self.correlationsIndices = None
        self.boolNegativeTau = False
        # ## plotting booleans and parameters
        self.boolPlotData = False
        self.boolPlotAverages = False
        self.boolPlotHamiltonian = False
        self.boolPlotDMAnimation = False
        self.boolPlotDMRedAnimation = False
        self.boolPlotOccEn = False
        self.boolPlotOccEnDiag = False
        self.boolPlotOccEnDiagExp = False
        self.boolPlotOffDiagOcc = False
        self.boolPlotOffDiagOccSingles = False
        self.boolPlotOffDiagDens = False
        self.boolPlotOffDiagDensRed = False
        self.boolPlotEngy = False
        self.boolPlotDecomp = False
        self.boolPlotTimescale = False
        self.boolPlotDOS = False
        self.boolPlotSpectralDensity = False
        self.boolPlotGreen = False
        # ## plotting variables
        self.dmFilesStepSize = 0
        self.dmFilesFPS = 0
        self.plotFontSize = 0
        self.plotLegendSize = 0
        self.plotSavgolFrame = 0
        self.plotSavgolOrder = 0
        self.plotLoAvgPerc = 0.0
        self.plotTimeScale = 0.0
        self.greenStepDist = 0
        self.evolStepDist = 0
        self.dmFiles = 0
        # end of the list
        # a new list of file variables which also have to be initialized
        self.filEnt = None
        self.filEntSpec = None
        self.filNorm = None
        self.filOcc = None
        self.filTotEnt = None
        self.filTotalEnergy = None
        self.filRedEnergy = None
        self.filOffDiagOcc = None
        self.filOffDiagOccSingles = None
        self.filOffDiagDens = None
        self.filOffDiagDensRed = None
        self.filProg = None
        self.filGreen = None
        # end of the list

        self.loadConfig()
        if not plotOnly:
            prepFolders(0, self.boolDMStore, (self.boolDMRedStore or self.boolDMRedDiagStore),
                        self.boolOccEnStore or self.boolOccEnDiag,
                        self.boolEntanglementSpectrum, self.boolCleanFiles, self.dataFolder)
            # touch the progress file
            open(self.dataFolder + 'progress.log', 'w').close()
        # mask selects only not traced out states
        self.mask = np.ones(self.m, dtype=bool)
        for k in self.kRed:
            self.mask[k] = False
        self.systemLevelIterator = np.linspace(0, self.m - 1, self.m)[self.mask]
        self.dim = dimOfBasis(self.N, self.m)  # dimension of basis
        # system variables
        if not plotOnly:
            self.datType = dtType
            if self.datType is np.complex128:
                self.mpRawArrayType = 'd'
            elif self.datType is np.complex64:
                self.mpRawArrayType = 'f'
            else:
                exit('Data type %s not supported' % str(self.datType))
            self.basis = np.zeros((self.dim, self.m), dtype=np.int)
            fillBasis(self.basis, self.N, self.m)
            self.basisDict = basis2dict(self.basis, self.dim)
            self.state_mpRawArray = mpRawArray(self.mpRawArrayType, self.dim * 2)
            self.state = np.frombuffer(self.state_mpRawArray, dtype=self.datType).reshape((self.dim,))
            np.copyto(self.state, np.zeros(self.dim, dtype=self.datType))
            self.stateConj_mpRawArray = mpRawArray(self.mpRawArrayType, self.dim * 2)
            self.stateConj = np.frombuffer(self.stateConj_mpRawArray, dtype=self.datType).reshape((self.dim,))
            np.copyto(self.stateConj, np.zeros(self.dim, dtype=self.datType))
            # parameter for storing in file
            self.stateNorm = 0
            self.stateNormAbs = 0
            self.stateNormCheck = 1e1  # check if norm has been supressed too much
            # do not initialize yet - wait until hamiltonian decomposition has been done for memory efficiency
            self.densityMatrix_mpRawArray = mpRawArray(self.mpRawArrayType, 0)
            self.densityMatrix = np.array([])
            self.densityMatrixInd = False
            self.multiprocDic = {}  # needed for multiprocessing
            self.entropy = 0
            self.totalEnergy = 0
            if self.boolReducedEnergy:
                self.reducedEnergy = 0
            self.operators = quadraticArray(self)
            self.occNo = np.zeros(self.m, dtype=np.float64)
            # hamiltonian - initialized with zeros (note - datatype is not! complex)
            self.hamiltonian = coo_matrix(np.zeros((self.dim, self.dim)), shape=(self.dim, self.dim),
                                          dtype=np.float64).tocsr()
            # matrix for time evolution - initially empty
            self.evolutionMatrix = None
            # eigenvalue and vectors
            self.eigVals = np.array([])
            self.eigVects = np.array([])
            self.eigInd = False

            # iteration step
            self.evolStep = 0
            self.evolStepTmp = 0
            self.evolTime = 0
            self.tavg = 0  # needed for estimation of remaining time
            self.dmcount = 0  # needed for numbering of density matrix files
            self.dmFileFactor = 0  # counting value for density matrix storage
            # ##### variables for the partial trace algorithm
            self.mRed = self.m - len(self.kRed)  # the number of leftover levels
            self.mRedComp = len(self.kRed)  # the number of traced out levels
            self.entropyRed = 0

            # ##### energy eigenbasis stuff
            if self.boolOffDiagOcc or self.boolDiagExpStore or self.boolOccEnStore or self.boolOccEnDiag:
                self.enState = np.zeros(self.dim, dtype=self.datType)
                self.offDiagOccMat = np.empty(self.m, dtype=object)
            if self.boolOffDiagOcc:
                self.offDiagOcc = np.zeros(self.m, dtype=self.datType)
                if self.occEnSingle > 0:
                    self.occEnInds = np.zeros((self.m, 2, self.occEnSingle), dtype=np.uint16)
                    self.offDiagOccSingles = np.zeros((self.m, self.occEnSingle), dtype=self.datType)

            if self.boolOffDiagDens:
                self.offDiagDens = 0
                self.eigInv = np.array([])
            # NOTE reduced density matrix offdiagonal stuff is initiated AFTER the reduced basis is created!

            if self.mRedComp == 0:
                self.dimRed = 0
                self.offsetsRed = None
                self.basisRed = None
                self.dimRedComp = self.dim
                self.offsetsRedComp = np.zeros((self.N + 2), dtype=np.int32)
                self.offsetsRedComp[-1] = self.dim
                self.basisRedComp = self.basis
                self.densityMatrixRed = None
            else:
                # particle number bound from above -> dim is that of 1 state more but with particle number conservation
                self.dimRed = dimOfBasis(self.N, (self.mRed + 1))
                self.offsetsRed = basisOffsets(self.N, self.mRed)
                self.basisRed = np.zeros((self.dimRed, self.mRed), dtype=np.int)
                fillReducedBasis(self.basisRed, self.N, self.mRed, self.offsetsRed)
                self.basisDictRed = basis2dict(self.basisRed, self.dimRed)  # only!! needed for reduced space operators
                self.dimRedComp = dimOfBasis(self.N, (self.mRedComp + 1))
                self.offsetsRedComp = basisOffsets(self.N, self.mRedComp)
                self.basisRedComp = np.zeros((self.dimRedComp, self.mRedComp), dtype=np.int)
                fillReducedBasis(self.basisRedComp, self.N, self.mRedComp, self.offsetsRedComp)
                self.reductionSteps = self.calculateReductionSteps()
                if not (self.m == 2 and self.mRed == 1):
                    # note that the iterator is not defined if not necessary
                    if self.reductionSteps < 4e4 and not cores:
                        self.cores = 1
                    self.iteratorRed = np.array_split(self.generateIteratorRed(), self.cores)
                    self.densityMatrixRedCoordinates = np.concatenate(
                        [np.concatenate(self.iteratorRed).T[:2].T, np.concatenate(self.iteratorRed).T[1::-1].T]
                    ).T
                # if only one level left, reduced matrix is diagonal
                if self.mRed == 1:
                    self.densityMatrixRed = np.zeros((self.dimRed,), dtype=self.datType)
                else:
                    # if more than one level left, use multiprocessing
                    # a shared matrix is not quite necessary by construction but do it anyways
                    self.densityMatrixRed_mpRawArray = mpRawArray(self.mpRawArrayType, self.dimRed ** 2 * 2)
                    self.densityMatrixRed = \
                        np.frombuffer(self.densityMatrixRed_mpRawArray,
                                      dtype=self.datType).reshape((self.dimRed, self.dimRed))
                    self.densityMatrixRedConstructorRaw = {}  # is a dictionary
                    self.densityMatrixRedConstructor = np.empty((self.cores,), dtype=np.ndarray)
                    self.densityMatrixRedConstructorConjRaw = {}  # is a dictionary
                    self.densityMatrixRedConstructorConj = np.empty((self.cores,), dtype=np.ndarray)
                    i = 0
                    # fill dictionary by process number, each gets an array
                    for el in self.iteratorRed:
                        self.densityMatrixRedConstructorRaw["%i" % i] = mpRawArray(self.mpRawArrayType, len(el) * 2)
                        self.densityMatrixRedConstructor[i] = \
                            np.frombuffer(self.densityMatrixRedConstructorRaw["%i" % i],
                                          dtype=self.datType).reshape((len(el),))
                        self.densityMatrixRedConstructorConjRaw["%i" % i] = mpRawArray(self.mpRawArrayType, len(el) * 2)
                        self.densityMatrixRedConstructorConj[i] = \
                            np.frombuffer(self.densityMatrixRedConstructorRaw["%i" % i],
                                          dtype=self.datType).reshape((len(el),))
                        i += 1
            if self.boolDMRedDiagStore:
                self.densityMatrixRedEigenvalues = None
                self.densityMatrixRedEigenvectors = None

            # NOTE here is the density matrix offdiagonal stuff
            if self.boolOffDiagDensRed:
                self.offDiagDensRed = 0
            if self.boolOffDiagDensRed or self.boolReducedEnergy or self.boolEntanglementSpectrum:
                self.hamiltonianRed = coo_matrix(np.zeros((self.dimRed, self.dimRed)),
                                                 shape=(self.dimRed, self.dimRed),
                                                 dtype=np.float64).tocsr()
                self.operatorsRed = quadraticArrayRed(self)
                self.eigValsRed = np.array([])
                self.eigVectsRed = np.array([])
                self.eigInvRed = np.array([])
                self.eigIndRed = False
            self.entanglementSpectrumIndicator = False
            self.entanglementSpectrum = np.zeros(self.dimRed)
            if self.boolEntanglementSpectrum:
                self.entanglementSpectrumEnergy = np.zeros(self.dimRed)
                self.systemOccupationOperator = self.operatorsRed[0, 0]
                for i in range(1, self.mRed):
                    self.systemOccupationOperator += self.operatorsRed[i, i]
                self.reducedHamiltonian = np.zeros(self.dimRed)
                self.reducedHamiltonianInd = False
            # ## Spectral
            if self.boolRetgreen:
                # lo
                self.greenLesserDim = dimOfBasis(self.N - 1, self.m)
                self.greenLesserBasis = np.zeros((self.greenLesserDim, self.m), dtype=np.int)
                fillBasis(self.greenLesserBasis, self.N - 1, self.m)
                self.greenLesserBasisDict = basis2dict(self.greenLesserBasis, self.greenLesserDim)
                self.greenLesserHamiltonian = coo_matrix(np.zeros((self.greenLesserDim, self.greenLesserDim)),
                                                         shape=(self.greenLesserDim, self.greenLesserDim),
                                                         dtype=np.float64).tocsr()
                self.greenLesserEigVals = np.array([])
                self.greenLesserEigVects = np.array([])
                self.greenLesserEigInd = False
                # hi
                self.greenGreaterDim = dimOfBasis(self.N + 1, self.m)
                self.greenGreaterBasis = np.zeros((self.greenGreaterDim, self.m), dtype=np.int)
                fillBasis(self.greenGreaterBasis, self.N + 1, self.m)
                self.greenGreaterBasisDict = basis2dict(self.greenGreaterBasis, self.greenGreaterDim)
                self.greenGreaterHamiltonian = coo_matrix(np.zeros((self.greenGreaterDim, self.greenGreaterDim)),
                                                          shape=(self.greenGreaterDim, self.greenGreaterDim),
                                                          dtype=np.float64).tocsr()
                self.greenGreaterEigVals = np.array([])
                self.greenGreaterEigVects = np.array([])
                self.greenGreaterEigInd = False
                # general
                self.green = np.zeros(self.m, dtype=self.datType)
                self.stateSaves = []  # append with time dep. state vector
                self.timeSaves = []  # append with time of saved state vector
                self.greenLesserEvolutionMatrix = None
                self.greenGreaterEvolutionMatrix = None
                self.greenLesserAnnihilation = []
                self.greenGreaterCreation = []
                # fill'em
                for i in range(self.m):
                    # note that the lowering operator transposed is the raising op. of the lower dimension space
                    self.greenLesserAnnihilation.append(getLesserAnnihilation(self, i))
                    # the raising operator transposed is the lowering op. of the higher dimension space
                    self.greenGreaterCreation.append(getGreaterCreation(self, i))

        self.openProgressFile()
        self.filProg.write('Initialized the class in ' + time_elapsed(t0, 60, 0) + '\n')
        self.closeProgressFile()
        print('Initialized the class in ' + time_elapsed(t0, 60, 0) + '\n')

    # end of init

    ###### reading from config file
    def loadConfig(self):
        configParser = configparser.RawConfigParser()
        # read the defaults and look for it in existing folder or parent folder
        if os.path.isfile('./default.ini'):
            configParser.read('./default.ini')
        elif os.path.isfile('../default.ini'):
            configParser.read('../default.ini')
        else:
            exit('Unable to read default.ini')
        # read the actual config file
        configParser.read('./' + self.confFile)
        # ## system parameters
        self.N = int(configParser.getfloat('system', 'N'))
        self.m = int(configParser.getfloat('system', 'm'))
        self.kRed = configParser.get('system', 'kred').split(',')
        if self.kRed[0] == '':
            self.kRed = np.array([])
        else:
            self.kRed = [int(el) for el in self.kRed]
        # ## hamiltonian parameters
        self.onsite = np.float64(configParser.getfloat('hamiltonian', 'de'))
        self.hybrid = np.float64(configParser.getfloat('hamiltonian', 't'))
        self.interequal = np.float64(configParser.getfloat('hamiltonian', 'ueq'))
        self.interdiff = np.float64(configParser.getfloat('hamiltonian', 'udiff'))
        # ## iteration parameters
        self.steps = int(configParser.getfloat('iteration', 'steps'))
        self.deltaT = np.float64(configParser.getfloat('iteration', 'deltaT'))
        self.order = int(configParser.getfloat('iteration', 'order'))
        self.loOrder = int(configParser.getfloat('iteration', 'loorder'))
        self.hiOrder = int(configParser.getfloat('iteration', 'hiorder'))
        self.deltaTgreen = np.float64(configParser.getfloat('iteration', 'deltaTgreen'))
        self.deltaTaugreen = np.float64(configParser.getfloat('iteration', 'deltaTaugreen'))
        self.greensteps = int(configParser.getfloat('iteration', 'greensteps'))

        # ## file management
        self.dataPoints = int(configParser.getfloat('filemanagement', 'datapoints'))
        self.boolDataStore = configParser.getboolean('filemanagement', 'datastore')
        self.boolClear = configParser.getboolean('filemanagement', 'clear')
        self.boolCleanFiles = configParser.getboolean('filemanagement', 'cleanfiles')
        self.dmFilesSkipFactor = int(configParser.getfloat('filemanagement', 'dmfile_skipfactor'))
        self.boolDMStore = configParser.getboolean('filemanagement', 'dmstore')
        self.boolDMRedStore = configParser.getboolean('filemanagement', 'dmredstore')
        self.boolDMRedDiagStore = configParser.getboolean('filemanagement', 'dmreddiag')
        self.boolHamilStore = configParser.getboolean('filemanagement', 'hamilstore')
        self.boolOccEnStore = configParser.getboolean('filemanagement', 'occenstore')
        self.boolOccEnDiag = configParser.getboolean('filemanagement', 'occendiag')
        self.boolDiagExpStore = configParser.getboolean('filemanagement', 'occendiagexp')
        self.boolOffDiagOcc = configParser.getboolean('filemanagement', 'offdiagocc')
        self.occEnSingle = configParser.getint('filemanagement', 'offdiagoccsingles')
        self.boolOffDiagDens = configParser.getboolean('filemanagement', 'offdiagdens')
        self.boolOffDiagDensRed = configParser.getboolean('filemanagement', 'offdiagdensred')
        self.boolEngyStore = configParser.getboolean('filemanagement', 'energiesstore')
        self.boolDecompStore = configParser.getboolean('filemanagement', 'decompstore')
        self.boolRetgreen = configParser.getboolean('filemanagement', 'retgreen')
        self.dataFolder = configParser.get('filemanagement', 'datafolder')
        if not self.dataFolder[-1] == '/':
            self.dataFolder = self.dataFolder + '/'
        # ## calculation-parameters
        self.boolOnlyRed = configParser.getboolean('calcparams', 'onlyreduced')
        self.boolTotalEntropy = configParser.getboolean('calcparams', 'totalentropy')
        self.boolReducedEntropy = configParser.getboolean('calcparams', 'reducedentropy')
        self.boolTotalEnergy = configParser.getboolean('calcparams', 'totalenergy')
        self.boolReducedEnergy = configParser.getboolean('calcparams', 'reducedenergy')
        self.boolOccupations = configParser.getboolean('calcparams', 'occupations')
        self.boolEntanglementSpectrum = configParser.getboolean('calcparams', 'entanglementspectrum')
        self.correlationsIndices = configParser.get('calcparams', 'correlations').split(';')
        if self.correlationsIndices[0] == '':
            self.correlationsIndices = np.array([])
        else:
            self.correlationsIndices = [[int(single) for single in el.split(',')] for el in self.correlationsIndices]
            # check if input is valid
            i = 0
            while i < len(self.correlationsIndices):
                if any(x < 0 for x in self.correlationsIndices[i]) or \
                        any(x > (self.m - 1) for x in self.correlationsIndices[i]):
                    print("ERROR: Following correlation indices are out of bounds and thus omitted:",
                          self.correlationsIndices.pop(i))
                    i -= 1
                elif len(self.correlationsIndices[i]) % 2 != 0:
                    print("ERROR: Following correlation indices are not quadratic and thus omitted:",
                          self.correlationsIndices.pop(i))
                    i -= 1
                i += 1
        self.boolNegativeTau = configParser.getboolean('calcparams', 'negativetau')
        if self.boolNegativeTau:
            print("WARNING - negative Tau is only working for greenOnly ríght now")
        # ## plotting booleans and parameters
        self.boolPlotData = configParser.getboolean('plotbools', 'data')
        self.boolPlotAverages = configParser.getboolean('plotbools', 'averages')
        self.boolPlotHamiltonian = configParser.getboolean('plotbools', 'hamiltonian')
        self.boolPlotDMAnimation = configParser.getboolean('plotbools', 'densistymatrix')
        self.boolPlotDMRedAnimation = configParser.getboolean('plotbools', 'reducedmatrix')
        self.boolPlotOccEn = configParser.getboolean('plotbools', 'occen')
        self.boolPlotOccEnDiag = configParser.getboolean('plotbools', 'occendiag')
        self.boolPlotOccEnDiagExp = configParser.getboolean('plotbools', 'occendiagexp')
        self.boolPlotOffDiagOcc = configParser.getboolean('plotbools', 'offdiagocc')
        self.boolPlotOffDiagOccSingles = configParser.getboolean('plotbools', 'offdiagoccsingles')
        self.boolPlotOffDiagDens = configParser.getboolean('plotbools', 'offdiagdens')
        self.boolPlotOffDiagDensRed = configParser.getboolean('plotbools', 'offdiagdensred')
        self.boolPlotEngy = configParser.getboolean('plotbools', 'energies')
        self.boolPlotDecomp = configParser.getboolean('plotbools', 'decomposition')
        self.boolPlotTimescale = configParser.getboolean('plotbools', 'timescale')
        self.boolPlotDOS = configParser.getboolean('plotbools', 'dos')
        self.boolPlotSpectralDensity = configParser.getboolean('plotbools', 'spectraldensity')
        self.boolPlotGreen = configParser.getboolean('plotbools', 'green')

        # ## plotting variables
        self.dmFilesStepSize = configParser.getint('plotvals', 'dmstepsize')
        self.dmFilesFPS = configParser.getint('plotvals', 'dmfps')
        self.plotFontSize = configParser.getint('plotvals', 'fontsize')
        self.plotLegendSize = configParser.getint('plotvals', 'legendsize')
        self.plotSavgolFrame = configParser.getint('plotvals', 'avg_frame')
        self.plotSavgolOrder = configParser.getint('plotvals', 'avg_order')
        self.plotLoAvgPerc = configParser.getfloat('plotvals', 'loavgperc') / 100.0
        # normally some coefficient in the hamiltonian (J or t)
        self.plotTimeScale = configParser.getfloat('plotvals', 'timescale')

        # the green function time step size
        if self.deltaTgreen != 0 and self.deltaTaugreen != 0:
            self.greenStepDist = int(round(self.deltaTaugreen / self.deltaTgreen))
            if self.deltaTaugreen % self.deltaTgreen >= 1:
                print("WARNING: There was a residue while dividing deltaTaugreen by deltaTgreen!")
            if self.greenStepDist % 2 != 0:
                print("WARNING: The green step distance is not dividable by two without residue!"
                      " This might cause problems in the green function calculation. It is: %f" % self.greenStepDist)

        # calculate the time steps that are taken at once
        self.evolStepDist = int(self.steps / self.dataPoints)
        if self.evolStepDist < 100:
            self.steps = 100 * self.dataPoints
            self.evolStepDist = 100
            print(
                'Number of steps must be at least factor 100 larger than datapoints! New number of steps: %e' % self.steps)
        self.dmFiles = self.dataPoints / self.dmFilesSkipFactor

        if self.dataPoints > self.steps:
            self.dataPoints = self.steps / 100
            print(
                'Number of data points was larger than number of steps - think again! Fixed the number of data points to be: %e' % self.dataPoints)

    ###### Methods:
    def updateDensityMatrix(self):
        if not self.densityMatrixInd:
            self.densityMatrix_mpRawArray = mpRawArray(self.mpRawArrayType, self.dim ** 2 * 2)
            self.densityMatrix = \
                np.frombuffer(self.densityMatrix_mpRawArray, dtype=self.datType).reshape((self.dim, self.dim))
            np.copyto(self.densityMatrix, np.zeros((self.dim, self.dim), dtype=self.datType))
            self.densityMatrixInd = True

        np.outer(self.state, self.state.conj(), self.densityMatrix)

    # end of updateDensityMatrix

    def calculateReductionSteps(self):
        # calculate the number of entries - it is always the upper triangle including the diagonal of the N-i particle
        # subspace multiplied with the dimension of the i particle complementary subspace
        entries = 0
        for i in range((self.N + 1)):
            dimred = dimOfBasis(self.N - i, self.mRed)
            entries += np.uint32((dimred ** 2 + dimred) / 2 * dimOfBasis(i, self.mRedComp))
        return entries

    def generateIteratorRed(self):
        # temporary basis vector occupation number containers
        el1 = np.zeros(self.m, dtype=np.int)
        el2 = np.zeros(self.m, dtype=np.int)

        iteratorRed = np.zeros((self.reductionSteps, 4), np.uint32)
        counter = 0
        # building up the block diagonal reduced matrix with the largest block first, N particles in sub-system
        for i in reversed(range(self.N + 1)):
            # go through all reduced basis vectors with i particles in it (note: offsetsRed is ascending)
            # reminder: offsetsRed is necesarrily always 0 for index N and dimRed on index N+1
            # j this is the row of the reduced matrix
            for j in range(self.offsetsRed[i], self.offsetsRed[i - 1]):
                # now go through all upper triangle (incl. diagonal) elements
                # partial trace algorithm is symmetric, regardless of matrix
                for jj in range(j, self.offsetsRed[i - 1]):
                    for k in range(self.offsetsRedComp[self.N - i], self.offsetsRedComp[self.N - i - 1]):
                        el1[self.mask] = self.basisRed[j]
                        el1[~self.mask] = self.basisRedComp[k]
                        el2[self.mask] = self.basisRed[jj]
                        el2[~self.mask] = self.basisRedComp[k]
                        iteratorRed[counter] = np.array(
                            [j, jj, self.basisDict[tuple(el1)], self.basisDict[tuple(el2)]])
                        counter += 1
        return iteratorRed

    # end of initIteratorRed

    def reduceDensityMatrix(self):
        if self.densityMatrixRed is None:
            return

        if self.mRed == 1 and self.m == 2:
            np.copyto(self.densityMatrixRed, self.densityMatrix.diagonal())
            return

        self.densityMatrixRed.fill(0)

        if self.cores == 1:
            initWorker(self.state_mpRawArray, self.stateConj_mpRawArray, self.densityMatrix_mpRawArray,
                       self.dim, self.dimRed, self.datType, self.densityMatrixRedConstructorRaw,
                       self.densityMatrixRedConstructorConjRaw)
            if self.mRed == 1 and self.m != 2:
                reduceDensityMatrix_mp_helper_small(self.iteratorRed[0], 0)
            else:
                reduceDensityMatrix_mp_helper(self.iteratorRed[0], 0)
        else:
            with Pool(processes=self.cores, initializer=initWorker, initargs=(
                            self.state_mpRawArray, self.stateConj_mpRawArray, self.densityMatrix_mpRawArray,
                            self.dim, self.dimRed, self.datType, self.densityMatrixRedConstructorRaw,
                            self.densityMatrixRedConstructorConjRaw)) as pool:
                if self.mRed == 1 and self.m != 2:
                    pool.map_async(reduceDensityMatrix_mp_helper_small,
                                   (self.iteratorRed, np.arange(self.cores)))
                else:
                    pool.map_async(reduceDensityMatrix_mp_helper,
                                   (self.iteratorRed, np.arange(self.cores)))

                pool.close()
                pool.join()

        coo_matrix(
            (np.concatenate([
                np.concatenate(self.densityMatrixRedConstructor), np.concatenate(self.densityMatrixRedConstructorConj)
            ]), self.densityMatrixRedCoordinates), shape=(self.dimRed, self.dimRed)
        ).toarray(out=self.densityMatrixRed)

    def reduceDensityMatrixFromState(self):
        if self.densityMatrixRed is None:
            return

        np.conjugate(self.state, out=self.stateConj)

        if self.mRed == 1 and self.m == 2:
            np.copyto(self.densityMatrixRed, np.abs(self.state) ** 2)
            return

        self.densityMatrixRed.fill(0)

        if self.cores == 1:
            initWorker(self.state_mpRawArray, self.stateConj_mpRawArray, self.densityMatrix_mpRawArray,
                       self.dim, self.dimRed, self.datType, self.densityMatrixRedConstructorRaw,
                       self.densityMatrixRedConstructorConjRaw)
            if self.mRed == 1 and self.m != 2:
                reduceDensityMatrixFromState_mp_helper_small(self.iteratorRed[0], 0)
            else:
                reduceDensityMatrixFromState_mp_helper(self.iteratorRed[0], 0)
        else:
            with Pool(processes=self.cores, initializer=initWorker, initargs=(
                            self.state_mpRawArray, self.stateConj_mpRawArray, self.densityMatrix_mpRawArray,
                            self.dim, self.dimRed, self.datType, self.densityMatrixRedConstructorRaw,
                            self.densityMatrixRedConstructorConjRaw)) as pool:
                if self.mRed == 1 and self.m != 2:
                    pool.map_async(reduceDensityMatrixFromState_mp_helper_small,
                                   (self.iteratorRed, np.arange(self.cores)))
                else:
                    pool.map_async(reduceDensityMatrixFromState_mp_helper,
                                   (self.iteratorRed, np.arange(self.cores)))

                pool.close()
                pool.join()

        coo_matrix(
            (np.concatenate([
                np.concatenate(self.densityMatrixRedConstructor), np.concatenate(self.densityMatrixRedConstructorConj)
            ]), self.densityMatrixRedCoordinates), shape=(self.dimRed, self.dimRed), dtype=self.datType
        ).toarray(out=self.densityMatrixRed)


    def reduceMatrix(self, matrx):
        tmpret = np.zeros((self.dimRed, self.dimRed), dtype=self.datType)
        if self.m == 2 and self.mRed == 1:
            for i in range(self.dimRed):
                tmpret[i, i] = matrx[i]
        else:
            for el in np.concatenate(self.iteratorRed):
                tmpret[el[0], el[1]] += matrx[el[2], el[3]]
                if el[0] != el[1]:
                    tmpret[el[1], el[0]] += matrx[el[3], el[2]]
        return tmpret

    # diagonalizes a reduced matrix respecting the block form, returns matrix of type sparse
    def diagonalizeReducedMatrix(self, reduced_matrix):
        if not np.allclose(reduced_matrix, reduced_matrix.T.conjugate()):
            print('Warning - reduced matrix for diagonalization is not hermitian')
            return None
        if self.dimRed == 1:
            if issparse(reduced_matrix):
                return reduced_matrix.toarray(), reduced_matrix
            else:
                return reduced_matrix, block_diag(reduced_matrix)

        block_matrix_array = []
        eigenvalue_array = np.array([])
        # start from N particle space to zero, just as in the reduction algorithm
        for i in reversed(range(self.N)):
            lo = self.offsetsRed[i]
            hi = self.offsetsRed[i - 1]
            tmpvals, tmpmat = la.eigh(reduced_matrix[lo:hi, lo:hi])
            block_matrix_array.append(tmpmat)
            eigenvalue_array = np.append(eigenvalue_array, tmpvals)
        return eigenvalue_array, block_diag(block_matrix_array)

    def diagonalizeReducedDensityMatrix(self):
        self.densityMatrixRedEigenvalues, self.densityMatrixRedEigenvectors = \
            self.diagonalizeReducedMatrix(self.densityMatrixRed)

    def saveReducedHamiltonian(self):
        redham = self.reduceMatrix(self.hamiltonian.toarray())
        save_npz(self.dataFolder + "reduced_hamiltonian.npz", coo_matrix(redham))
        tmpvals, tmpvects = self.diagonalizeReducedMatrix(redham)
        np.savetxt(self.dataFolder + "reduced_hamiltonian_eigenvalues.dat", tmpvals)
        save_npz(self.dataFolder + "reduced_hamiltonian_eigenvectors.npz", tmpvects)

    def initAllHamiltonians(self):
        t0 = tm.time()

        self.initHamiltonian()

        self.openProgressFile()
        self.filProg.write('Wrote hamiltonian in ' + time_elapsed(t0, 60, 0) + '\n')
        self.closeProgressFile()

        if self.boolOffDiagDensRed or self.boolReducedEnergy or self.boolEntanglementSpectrum:
            t1 = tm.time()

            self.initHamiltonianRed()

            self.openProgressFile()
            self.filProg.write('Wrote reduced hamiltonian in ' + time_elapsed(t1, 60, 0) + '\n')
            self.closeProgressFile()

        if self.boolRetgreen:
            t1 = tm.time()

            self.initgreenLesserHamiltonian()
            self.initgreenGreaterHamiltonian()

            self.openProgressFile()
            self.filProg.write('Wrote high and low hamiltonian in ' + time_elapsed(t1, 60, 0) + '\n')
            self.closeProgressFile()
        print('Wrote all hamiltonians in ' + time_elapsed(t0, 60, 0))

    # hamiltonian with equal index interaction different to non equal index interaction
    def initHamiltonian(self):
        for i in range(self.m):
            for j in range(self.m):
                if i != j:
                    self.hamiltonian += self.hybrid * self.operators[i, j]
                else:
                    self.hamiltonian += i * self.onsite * self.operators[i, j]

        if self.interequal != 0 and self.interdiff != 0:
            for i in range(self.m):
                for j in range(self.m):
                    for k in range(self.m):
                        for l in range(self.m):
                            tmp = getQuartic(self, i, j, k, l)
                            if i == j and k == l and k == j:
                                self.hamiltonian += self.interequal * tmp
                            else:
                                self.hamiltonian += self.interdiff * tmp
                            del tmp

    def initHamiltonianRed(self):
        for i in range(self.mRed):
            for j in range(self.mRed):
                if i != j:
                    self.hamiltonianRed += self.hybrid * self.operatorsRed[i, j]
                else:
                    self.hamiltonianRed += int(self.systemLevelIterator[i]) * self.onsite * self.operatorsRed[i, j]

        if self.interequal != 0 and self.interdiff != 0:
            for i in range(self.mRed):
                for j in range(self.mRed):
                    for k in range(self.mRed):
                        for l in range(self.mRed):
                            tmp = getQuarticRed(self, i, j, k, l)
                            if i == j and k == l and k == j:
                                self.hamiltonianRed += self.interequal * tmp
                            else:
                                self.hamiltonianRed += self.interdiff * tmp
                            del tmp

    def initgreenLesserHamiltonian(self):
        tmpgreenops = quadraticArrayGreenLesser(self)
        for i in range(self.m):
            for j in range(self.m):
                if i != j:
                    self.greenLesserHamiltonian += self.hybrid * tmpgreenops[i, j]
                else:
                    self.greenLesserHamiltonian += i * self.onsite * tmpgreenops[i, j]

        if self.interequal != 0 and self.interdiff != 0:
            for i in range(self.m):
                for j in range(self.m):
                    for k in range(self.m):
                        for l in range(self.m):
                            tmp = getQuarticGreen(tmpgreenops, i, j, k, l)
                            if i == j and k == l and k == j:
                                self.greenLesserHamiltonian += self.interequal * tmp
                            else:
                                self.greenLesserHamiltonian += self.interdiff * tmp
                            del tmp
            del tmpgreenops

    def initgreenGreaterHamiltonian(self):
        tmpgreenops = quadraticArrayGreenGreater(self)
        for i in range(self.m):
            for j in range(self.m):
                if i != j:
                    self.greenGreaterHamiltonian += self.hybrid * tmpgreenops[i, j]
                else:
                    self.greenGreaterHamiltonian += i * self.onsite * tmpgreenops[i, j]

        if self.interequal != 0 and self.interdiff != 0:
            for i in range(self.m):
                for j in range(self.m):
                    for k in range(self.m):
                        for l in range(self.m):
                            tmp = getQuarticGreen(tmpgreenops, i, j, k, l)
                            if i == j and k == l and k == j:
                                self.greenGreaterHamiltonian += self.interequal * tmp
                            else:
                                self.greenGreaterHamiltonian += self.interdiff * tmp
                            del tmp
            del tmpgreenops

    def initAllEvolutionMatrices(self):
        t0 = tm.time()

        self.initEvolutionMatrix()

        self.openProgressFile()
        self.filProg.write('Initialized evolution matrix in ' + time_elapsed(t0, 60, 0) + '\n')
        self.closeProgressFile()
        if self.boolRetgreen:
            t1 = tm.time()

            self.initgreenLesserEvolutionMatrix(conj=self.boolNegativeTau)
            self.initgreenGreaterEvolutionMatrix(conj=self.boolNegativeTau)

            self.openProgressFile()
            self.filProg.write('Initialized high and low evolution matrices in ' + time_elapsed(t1, 60, 0) + '\n')
            self.closeProgressFile()
        print('Initialized evolution matrices in ' + time_elapsed(t0, 60, 0))

    # The matrix already inherits the identity so step is just mutliplication
    # time evolution order given by order of the exponential series
    def initEvolutionMatrix(self, diagonalize=False, conj=False, greenstep=False):
        if self.order == 0:
            print('Warning - Time evolution of order 0 means no dynamics...')
        if not np.allclose(self.hamiltonian.toarray(), self.hamiltonian.toarray().T.conjugate()):
            print('Warning - hamiltonian is not hermitian!')
        if self.evolutionMatrix is None:
            # apparently C-order is faster when using the dot product with out
            self.evolutionMatrix = np.zeros((self.dim, self.dim), dtype=self.datType, order='C')
        tmpmat = spidentity(self.dim, dtype=self.datType, format='csr')

        if conj:
            pre = 1j
        else:
            pre = (-1j)

        # the normal evolution matrix
        if not greenstep:
            for i in range(1, self.order + 1):
                tmpmat += (pre ** i) * (self.deltaT ** i) * (self.hamiltonian ** i) / factorial(i)

            # for greenOnly, perform the WHOLE time evolution in one step (holy moly)
            # otherwise use the time step distance as prepared in __init__
            # note that it will be cast to c-contiguous order automatically
            if self.greenOnly:
                np.copyto(self.evolutionMatrix, npmatrix_power(tmpmat.toarray(order='F'), self.steps))
            else:
                np.copyto(self.evolutionMatrix, npmatrix_power(tmpmat.toarray(order='F'), self.evolStepDist))

            if self.boolHamilStore:
                storeMatrix(self.hamiltonian.toarray(), self.dataFolder + 'hamiltonian.dat', 1)
                storeMatrix(self.evolutionMatrix, self.dataFolder + 'evolutionmatrix.dat', 1)

            # Store hamiltonian eigenvalues
            if diagonalize:
                self.updateEigenenergies()
        # the short step evolution matrix for the green function calculation
        else:
            for i in range(1, self.order + 1):
                tmpmat += (pre ** i) * (self.deltaTgreen ** i) * (self.hamiltonian ** i) / factorial(i)
            np.copyto(self.evolutionMatrix, npmatrix_power(tmpmat.toarray(order='F'), int(self.greenStepDist / 2)))

    # end

    # The matrix already inherits the identity so step is just mutliplication
    # time evolution order given by order of the exponential series
    # this one will be only in sparse container since it is meant for sparse matrix mult.
    #### IMPORTANT NOTE - complex conjugate will be needed for Green function ####
    #### FURTHER: need only 2*delta_T for green function, so added sq=True ####
    #### ALSO: delta_T here is actually the delta_T of time-steps, so the wide steps!!! ####
    def initgreenLesserEvolutionMatrix(self, diagonalize=False, conj=False, sq=True):
        if self.loOrder == 0:
            print('Warning - Time evolution of order 0 means no dynamics...')
        if not np.allclose(self.greenLesserHamiltonian.toarray(), self.greenLesserHamiltonian.toarray().T.conj()):
            print('Warning - hamiltonian is not hermitian!')
        self.greenLesserEvolutionMatrix = spidentity(self.greenLesserDim, dtype=self.datType, format='csr')

        if conj:
            pre = 1j
        else:
            pre = (-1j)

        # normal green function evolution operator for deltaT
        if not self.greenOnly:
            for i in range(1, self.loOrder + 1):
                self.greenLesserEvolutionMatrix += (pre ** i) * (self.deltaT ** i) \
                                                   * (self.greenLesserHamiltonian ** i) / factorial(i)
            if diagonalize:
                self.updateLoEigenenergies()

            # bring it to the same timestep distance as the state vector
            self.greenLesserEvolutionMatrix = np.asfortranarray(
                npmatrix_power(self.greenLesserEvolutionMatrix.toarray(), self.evolStepDist), dtype=self.datType)
            if sq and not self.greenOnly:
                self.greenLesserEvolutionMatrix = npmatrix_power(self.greenLesserEvolutionMatrix, 2)
        # green only time evolution operator for deltaTgreen
        else:
            for i in range(1, self.loOrder + 1):
                self.greenLesserEvolutionMatrix += (pre ** i) * (self.deltaT ** i) * \
                                                   (self.greenLesserHamiltonian ** i) / factorial(i)
            # raise to appropriate power
            self.greenLesserEvolutionMatrix = np.asfortranarray(
                npmatrix_power(self.greenLesserEvolutionMatrix.toarray(), self.greenStepDist), dtype=self.datType)

    # end

    # The matrix already inherits the identity so step is just mutliplication
    # time evolution order given by order of the exponential series
    # this one will be only in sparse container since it is meant for sparse matrix mult.
    #### IMPORTANT NOTE - complex conjugate will be needed for Green function ####
    #### FURTHER: need only 2*delta_T for green function, so added sq=True ####
    #### ALSO: delta_T here is actually the delta_T of time-steps, so the wide steps!!! ####
    def initgreenGreaterEvolutionMatrix(self, diagonalize=False, conj=False, sq=True):
        if self.hiOrder == 0:
            print('Warning - Time evolution of order 0 means no dynamics...')
        if not np.allclose(self.greenGreaterHamiltonian.toarray(), self.greenGreaterHamiltonian.toarray().T.conj()):
            print('Warning - hamiltonian is not hermitian!')
        self.greenGreaterEvolutionMatrix = spidentity(self.greenGreaterDim, dtype=self.datType, format='csr')

        # in contrast to the greater evolution matrix the default here is positive exponent
        if conj:
            pre = (-1j)
        else:
            pre = 1j

        # normal green function evolution operator for deltaT
        if not self.greenOnly:
            for i in range(1, self.hiOrder + 1):
                self.greenGreaterEvolutionMatrix += (pre ** i) * (self.deltaT ** i) * \
                                                    (self.greenGreaterHamiltonian ** i) / factorial(i)
            if diagonalize:
                self.updateHiEigenenergies()
            self.greenGreaterEvolutionMatrix = np.asfortranarray(
                npmatrix_power(self.greenGreaterEvolutionMatrix.toarray(), self.evolStepDist), dtype=self.datType)
            if sq:
                self.greenGreaterEvolutionMatrix = npmatrix_power(self.greenGreaterEvolutionMatrix, 2)
        # green only time evolution operator for deltaTgreen
        else:
            for i in range(1, self.loOrder + 1):
                self.greenGreaterEvolutionMatrix += (pre ** i) * (self.deltaT ** i) \
                                                    * (self.greenGreaterHamiltonian ** i) / factorial(i)
            # raise to appropriate power
            self.greenGreaterEvolutionMatrix = np.asfortranarray(
                npmatrix_power(self.greenGreaterEvolutionMatrix.toarray(), self.greenStepDist), dtype=self.datType)

    # end

    def greenStoreState(self):
        self.stateSaves.append(self.state)
        self.timeSaves.append(self.evolTime)

    # approximate distributions in energy space - all parameters have to be set!
    # if skip is set to negative, the absolute value gives probability for finding a True in binomial
    def stateEnergy(self, muperc=None, sigma=None, phase=None, skip=None, dist=None, peakamps=None, skew=None):
        if muperc is None:
            muperc = [50]
        if sigma is None:
            sigma = [1]
        if phase is None:
            phase = ['none']
        if skip is None:
            skip = [0]
        if dist is None:
            dist = ['std']
        if peakamps is None:
            peakamps = [1]
        if skew is None:
            skew = [0]

        # initialize everything
        tmpdist = None
        phaseArray = np.array([])
        skipArray = np.array([])
        # end of initializing

        if not self.eigInd:
            self.updateEigenenergies()

        self.state.fill(0)

        for i in range(len(muperc)):
            if dist[i] == 'std':
                dind = 1
            elif dist[i] == 'rect':
                dind = 2
            elif dist[i] == 'rnd':
                dind = 3
                tmpdist = np.random.rand(self.dim)
            else:
                dind = 1

            if phase[i] == 'none':
                phaseArray = np.zeros(self.dim)
            elif phase[i] == 'alt':
                phaseArray = np.zeros(self.dim)
                phaseArray[::2] = np.pi
            elif phase[i] == 'rnd':
                phaseArray = np.random.rand(self.dim) * 2 * np.pi
            elif phase[i] == 'rndreal':
                phaseArray = np.random.binomial(1, 0.5, self.dim) * np.pi
            else:
                phaseArray = np.zeros(self.dim)

            if skip[i] < 0:
                skipArray = np.random.binomial(1, -1 * skip[i], self.dim)
            elif skip[i] == 0:
                skipArray = np.zeros(self.dim)
                skipArray[::1] = 1
            else:
                skipArray = np.zeros(self.dim)
                skipArray[::int(skip[i])] = 1

            # mu is given in percent so get mu in energy space - also offsets are taken into account
            mu = self.eigVals[0] + (muperc[i] / 100) * (self.eigVals[-1] - self.eigVals[0])
            for k in range(self.dim):
                if skipArray[k]:
                    if dind == 1:
                        self.state += peakamps[i] * np.exp(1j * phaseArray[k]) * gaussian(self.eigVals[k], mu, sigma[i],
                                                                                          norm=True,
                                                                                          skw=skew[i]) * self.eigVects[
                                                                                                         :, k]
                    elif dind == 2:
                        self.state += peakamps[i] * np.exp(1j * phaseArray[k]) * rect(self.eigVals[k], mu, sigma[i],
                                                                                      norm=False) * self.eigVects[:, k]
                    elif dind == 3:
                        self.state[k] += peakamps[i] * np.exp(1j * phaseArray[k]) * tmpdist[k]
        del phaseArray
        del skipArray
        self.normalize(True)

    ### This one is tailored for taking exactly one energy as the weight
    # approximate distributions in energy space - all parameters have to be set!
    # if skip is set to negative, the absolute value gives probability for finding a True in binomial
    def stateEnergyMicrocan(self, avgen=0, sigma=1, phase='none', skip=0, dist='rect', peakamps=1, skew=0):
        if not self.eigInd:
            self.updateEigenenergies()

        # initialize everything
        tmpdist = None
        # end of initializing

        self.state.fill(0)

        if dist == 'std':
            dind = 1
        elif dist == 'rect':
            dind = 2
        elif dist == 'rnd':
            dind = 3
            tmpdist = np.random.rand(self.dim)
        else:
            dind = 1

        if phase == 'none':
            phaseArray = np.zeros(self.dim)
        elif phase == 'alt':
            phaseArray = np.zeros(self.dim)
            phaseArray[::2] = np.pi
        elif phase == 'rnd':
            phaseArray = np.random.rand(self.dim) * 2 * np.pi
        elif phase == 'rndreal':
            phaseArray = np.random.binomial(1, 0.5, self.dim) * np.pi
        else:
            phaseArray = np.zeros(self.dim)

        if skip < 0:
            skipArray = np.random.binomial(1, -1 * skip, self.dim)
        elif skip == 0:
            skipArray = np.zeros(self.dim)
            skipArray[::1] = 1
        else:
            skipArray = np.zeros(self.dim)
            skipArray[::int(skip)] = 1

        for k in range(self.dim):
            if skipArray[k]:
                if dind == 1:
                    self.state[:] += peakamps * np.exp(1j * phaseArray[k]) * gaussian(self.eigVals[k], avgen, sigma,
                                                                                      norm=True,
                                                                                      skw=skew) * self.eigVects[:, k]
                elif dind == 2:
                    self.state[:] += peakamps * np.exp(1j * phaseArray[k]) * rect(self.eigVals[k], avgen, sigma,
                                                                                  norm=False) * self.eigVects[:, k]
                elif dind == 3:
                    self.state[k] += peakamps * np.exp(1j * phaseArray[k]) * tmpdist[k]
        del phaseArray
        del skipArray
        self.normalize(True)

    def normalize(self, initial=False):
        # note that the shape of the state vector is (dim,1) for reasons of matrix multiplication in numpy
        self.stateNorm = la.norm(self.state)
        self.stateNormAbs *= self.stateNorm
        self.state /= self.stateNorm
        # do not store the new state norm - it is defined to be 1 so just store last norm value!
        # self.stateNorm = np.real(sqrt(npeinsum('ij,ij->j',self.state,np.conjugate(self.state))))[0]
        if bool(initial):
            self.stateNormAbs = 1
            self.updateEigendecomposition()
            # store starting states used for green function
        if np.abs(self.stateNormAbs) > self.stateNormCheck:
            if self.stateNormCheck == 1e1:
                print(
                    '\n' + '### WARNING! ### state norm has been normalized by more than the factor 10 now!' + '\n' + 'Check corresponding plot if behavior is expected - indicator for numerical instability!' + '\n')
                self.stateNormCheck = 1e2
            else:
                self.closeFiles()
                self.plot()
                exit(
                    '\n' + 'Exiting - state norm has been normalized by more than the factor 100, numerical error is very likely.')

    # end of normalize

    # note that - in principle - the expectation value can be complex! (though it shouldn't be)
    # this one explicitly uses the state vector - note that this also works with sparse matrices!
    def expectValue(self, operator):
        if operator.shape != (self.dim, self.dim):
            exit('Operator shape is' + str(np.shape(operator)) + 'but' + str((self.dim, self.dim)) + 'is needed!')
        # return multi_dot([np.conjugate(np.array(self.state)[:,0]), operator, np.array(self.state)[:,0]])
        return vdot(self.state, operator.dot(self.state))

    # note that - in principle - the expectation value can be complex! (though it shouldn't be)
    def expectValueDM(self, operator):
        if operator.shape != (self.dim, self.dim):
            exit('Operator shape is' + str(np.shape(operator)) + 'but' + str((self.dim, self.dim)) + 'is needed!')
        # this order is needed for sparse matrices - still works because of cyclic invariance
        return np.trace(operator.dot(self.densityMatrix))

    def expectValueRed(self, operator):
        if operator.shape != (self.dimRed, self.dimRed):
            exit('Operator shape is' + str(np.shape(operator)) + 'but' + str((self.dimRed, self.dimRed)) + 'is needed!')
        # this order is needed for sparse matrices - still works because of cyclic invariance
        if not self.mRed == 1:
            return np.trace(operator.dot(self.densityMatrixRed))
        else:
            # when the density matrix is diagonal, this is essentially the formula
            return np.diag(operator).dot(self.densityMatrixRed)

    def updateEigenenergies(self):
        if not self.eigInd:
            self.eigVals, self.eigVects = la.eigh(self.hamiltonian.toarray())
            self.eigInd = True

    def updateEigenenergiesRed(self):
        if not self.eigIndRed:
            self.eigValsRed, self.eigVectsRed = la.eigh(self.hamiltonianRed.toarray())
            self.eigInvRed = la.inv(np.matrix(self.eigVectsRed.T))
            self.eigIndRed = True
            # TMP TMP TMP
            tmpfil = open(self.dataFolder + 'eigenvaluesRed.dat', 'w')
            for i in range(self.dimRed):
                tmpfil.write("%i %.16e\n" % (i, self.eigValsRed[i]))
            tmpfil.close()

    def updateLoEigenenergies(self):
        if not self.greenLesserEigInd:
            self.greenLesserEigVals, self.greenLesserEigVects = la.eigh(self.greenLesserHamiltonian.toarray())
            self.greenLesserEigInd = True

    def updateHiEigenenergies(self):
        if not self.greenGreaterEigInd:
            self.greenGreaterEigVals, self.greenGreaterEigVects = la.eigh(self.greenGreaterHamiltonian.toarray())
            self.greenGreaterEigInd = True

            ## will free the memory!!!

    def updateEigendecomposition(self, clear=True):
        tmpmat = None
        eivectinv = None
        if self.boolEngyStore:
            t0 = tm.time()
            self.updateEigenenergies()
            self.openProgressFile()
            self.filProg.write('Hamiltonian diagonalized in ' + time_elapsed(t0, 60, 0) + '\n')
            self.closeProgressFile()
            print("Hamiltonian diagonalized in " + time_elapsed(t0, 60, 0))
            t0 = tm.time()

            # decomposition in energy space       
            tfil = open(self.dataFolder + 'hamiltonian_eigvals.dat', 'w')
            if self.boolDecompStore:
                tmpAbsSq = np.zeros(self.dim)
                # generate all overlaps at once
                # note that conj() is optional since the vectors can be chosen to be real
                self.enState = self.eigVects.T.conj().dot(self.state)

                # also calculate all occupation numbers at once
                enoccs = np.zeros((self.m, self.dim))
                for j in range(self.m):
                    enoccs[j] = np.diag(np.dot(self.eigVects.T.conj(), self.operators[j, j].dot(self.eigVects))).real

                for i in range(self.dim):
                    # absolute value of overlap
                    tmpAbsSq[i] = np.abs(self.enState[i]) ** 2
                    if tmpAbsSq[i] != 0:
                        # if nonzero we want to have the angle in the complex plane in units of two pi
                        tmpPhase = np.angle(self.enState[i]) / (2 * np.pi)
                    else:
                        tmpPhase = 0

                    # occupation numbers of the eigenvalues
                    tfil.write('%i %.16e %.16e %.16e ' % (i, self.eigVals[i], tmpAbsSq[i], tmpPhase))
                    for j in range(self.m):
                        tfil.write('%.16e ' % (enoccs[j, i]))
                    tfil.write('\n')
            else:
                for i in range(self.dim):
                    tfil.write('%i %.16e\n' % (i, self.eigVals[i]))
            tfil.close()

            # decomposition in fock space
            sfil = open(self.dataFolder + 'state.dat', 'w')
            for i in range(self.dim):
                tmpAbsSqFck = np.abs(self.state[i]) ** 2
                if tmpAbsSqFck != 0:
                    tmpPhase = np.angle(self.state[i]) / (2 * np.pi)  # angle in complex plane in units of two pi
                else:
                    tmpPhase = 0
                # occupation numbers of the eigenvalues
                sfil.write('%i %.16e %.16e ' % (i, tmpAbsSqFck, tmpPhase))
                for j in range(self.m):
                    sfil.write('%i ' % self.basis[i, j])
                sfil.write('\n')
            sfil.close()
            self.openProgressFile()
            self.filProg.write('Eigendecomposition completed in ' + time_elapsed(t0, 60, 0) + '\n')
            self.closeProgressFile()
            print("Eigendecomposition completed in " + time_elapsed(t0, 60, 0))
        if self.boolDiagExpStore or self.boolOccEnStore or self.boolOccEnDiag or self.boolOffDiagOcc or self.boolOffDiagDens:
            self.updateEigenenergies()
            eivectinv = la.inv(self.eigVects.T)  # this will later be stored in the class if needed

        # first store occupation matrices in energy basis if needed
        if self.boolDiagExpStore or self.boolOffDiagOcc or self.boolOccEnStore or self.boolOccEnDiag:
            for i in range(self.m):
                self.offDiagOccMat[i] = np.dot(self.eigVects.T, self.operators[i, i].dot(eivectinv))

        # expectation values in diagonal representation (ETH)
        if self.boolDiagExpStore or self.boolOffDiagOcc:
            t0 = tm.time()
            # if the state_energy_basis array has already been calculated this step can be omitted
            if not self.boolDecompStore:
                # generate all overlaps at once
                # note that conj() is optional since the vectors can be chosen to be real
                self.enState = self.eigVects.T.conj().dot(self.state)

            if self.boolDiagExpStore:
                # diagonals in expectation value    
                ethfil = open(self.dataFolder + 'occ_energybasis_diagonals_expectation.dat', 'w')
                for i in range(self.m):
                    tmpocc = np.dot(np.abs(self.enState) ** 2, self.offDiagOccMat[i].diagonal()).real
                    ethfil.write('%i %.16e \n' % (i, tmpocc))
                print("Occupation matrices transformed " + time_elapsed(t0, 60, 1))
                self.openProgressFile()
                self.filProg.write('Occupation matrices transformed in ' + time_elapsed(t0, 60, 1) + '\n')
                self.closeProgressFile()
                ethfil.close()

        # store the actual matrix to a file (might become very large!)
        if self.boolOccEnStore:
            t0 = tm.time()
            for i in range(self.m):
                # note that the off diag mat still contains the diagonals right now!
                storeMatrix(self.offDiagOccMat[i], self.dataFolder + 'occ_energybasis_%i.dat' % i, absOnly=0, stre=True,
                            stim=False,
                            stabs=False)
            print("Occupation number matrices stored in " + time_elapsed(t0, 60, 1))

        # store the matrix diagonals in a file
        if self.boolOccEnDiag:
            t0 = tm.time()
            head = "index energy matrix_element"
            head_weight = "index energy weighted_matrix_element"
            for i in range(self.m):
                # note that the off diag mat still contains the diagonals right now!
                # since the occupation numbers are real valued it is sufficient to only store the real part
                np.savetxt(self.dataFolder + 'occ_energybasis_diagonal_%i.dat' % i,
                           np.column_stack(
                               (
                                   np.arange(self.dim), self.eigVals, self.offDiagOccMat[i].diagonal().real
                               )
                           ), header=head)
                # store the matrix elements weighted by the state decompisitopn factor abs()^2
                np.savetxt(self.dataFolder + 'occ_energybasis_diagonal_weighted_%i.dat' % i,
                           np.column_stack(
                               (
                                   np.arange(self.dim), self.eigVals,
                                   # note: the following should multiply element-wise!
                                   self.offDiagOccMat[i].diagonal() * np.abs(self.enState) ** 2
                               )
                           ), header=head_weight)
            print("Occupation number matrix diagonals stored in " + time_elapsed(t0, 60, 1))

        # now we remove the diagonal elements
        if self.boolOffDiagOcc:
            for i in range(self.m):
                np.fill_diagonal(self.offDiagOccMat[i], 0)

        if self.occEnSingle and self.boolOffDiagOcc:
            t0 = tm.time()
            infofile = open(self.dataFolder + 'offdiagoccsinglesinfo.dat', 'w')
            if not (self.boolDiagExpStore or self.boolOffDiagOcc or self.boolDecompStore):
                # generate all overlaps at once
                # note that conj() is optional since the vectors can be chosen to be real
                self.enState = self.eigVects.T.conj().dot(self.state)

            # props to Warren Weckesser https://stackoverflow.com/questions/20825990/find-multiple-maximum-values-in-a-2d-array-fast
            # Get the indices for the largest `num_largest` values.
            num_largest = self.occEnSingle
            for i in range(self.m):
                # this is not optimized but one has to store it as a matrix for correct searching
                tmpmat = np.einsum('l,lj,j -> lj', self.enState.conj(), self.offDiagOccMat[i], self.enState,
                                   optimize=True)
                # tmpmat = np.outer(self.enState.conj(), np.dot(self.offDiagOccMat[i], self.enState))
                infofile.write('%i ' % i)
                # to use argpartition correctly we must treat the matrix as an array
                indices = tmpmat.argpartition(tmpmat.size - num_largest, axis=None)[-num_largest:]
                self.occEnInds[i, 0], self.occEnInds[i, 1] = np.unravel_index(indices, tmpmat.shape)
                for j in range(self.occEnSingle):
                    infofile.write('%i %i %.16e %16e ' % (
                        self.occEnInds[i, 0, j], self.occEnInds[i, 1, j], self.eigVals[self.occEnInds[i, 0, j]].real,
                        self.eigVals[self.occEnInds[i, 1, j]].real))
                infofile.write('\n')
            infofile.close()
            print("Largest elements found and infos stored in " + time_elapsed(t0, 60, 1))
            self.openProgressFile()
            self.filProg.write('Largest elements found and infos stored in ' + time_elapsed(t0, 60, 1) + '\n')
            self.closeProgressFile()

            del tmpmat  # not sure if this is neccessary but do it regardless...

        # store the inverted eigenvector matrix for later use
        if self.boolOffDiagDens:
            self.eigInv = eivectinv

        if clear:
            # free the memory
            del self.eigVals
            if not self.boolOffDiagOcc and not self.boolOffDiagDens:
                del self.eigVects
                self.eigVects = np.array([])
            self.eigVals = np.array([])
            self.eigInd = False

    def updateOffDiagOcc(self):
        # calculate all overlaps at once
        self.enState = np.dot(self.eigVects.T, self.state)

        for i in range(self.m):
            self.offDiagOcc[i] = vdot(self.enState, self.offDiagOccMat[i].dot(self.enState))

            # check for imaginary part -> would give an indication for errors
            if self.offDiagOcc[i].imag > 1e-6:
                print('The offdiagonal expectation value has an imaginary part of ', self.offDiagOcc[i].imag)

        if self.occEnSingle:
            for i in range(self.m):
                for j in range(self.occEnSingle):
                    x = int(self.occEnInds[i, 0, j])
                    y = int(self.occEnInds[i, 1, j])
                    self.offDiagOccSingles[i, j] = self.enState[x].conj() * self.offDiagOccMat[i][x, y] * self.enState[
                        y]

    def updateOffDiagDens(self):
        self.offDiagDens = (multi_dot(
            [np.ones(self.dim), self.eigVects.T, self.densityMatrix, self.eigInv, np.ones(self.dim)]) - np.trace(
            self.densityMatrix)).real

    def updateOffDiagDensRed(self):
        if self.mRed == 1:
            self.offDiagDensRed = 0 + 0j
        else:
            if not self.eigIndRed:
                self.updateEigenenergiesRed()
            self.offDiagDensRed = (multi_dot(
                [np.ones(self.dimRed), self.eigVectsRed.T, self.densityMatrixRed, self.eigInvRed,
                 np.ones(self.dimRed)]) - np.trace(self.densityMatrixRed)).real

    def updateEntropy(self):
        self.entropy = 0
        for el in la.eigvalsh(self.densityMatrix, check_finite=False):
            if el.real > 0:
                self.entropy -= el.real * nplog(el.real)
            if el.real < -1e-7:
                print('Oh god, there is a negative eigenvalue smaller than 1e-7 ! Namely:', el)

    # end of updateEntropy

    def updateEntanglementSpectrum(self):
        if self.mRed != 1:
            self.reduceDensityMatrixFromState()
            if self.boolEntanglementSpectrum and not self.entanglementSpectrumIndicator:
                if not self.reducedHamiltonianInd:
                    self.reducedHamiltonian = self.reduceMatrix(self.hamiltonian)
                self.entanglementSpectrum, entanglementStates = la.eigh(self.densityMatrixRed, check_finite=False)
                for i in range(self.dimRed):
                    '''
                    self.entanglementSpectrumEnergy[i] = \
                        np.vdot(entanglementStates[i],
                                self.hamiltonianRed.dot(entanglementStates[i])).real
                    '''
                    self.entanglementSpectrumEnergy[i] = \
                        vdot(entanglementStates[i],
                             self.hamiltonianRed.dot(entanglementStates[i])).real
                # this is the chemical potential search stuff!
                tmpinv = la.inv(np.matrix(entanglementStates.T))
                trans_hamil = \
                    np.dot(entanglementStates.T, self.reduceMatrix(self.hamiltonian).dot(tmpinv))
                trans_hamil -= np.identity(self.dimRed) * trans_hamil.diagonal()

            else:
                self.entanglementSpectrum = la.eigvalsh(self.densityMatrixRed, check_finite=False)
        else:
            # if only one level left, density matrix is already diagonal
            self.reduceDensityMatrixFromState()
            self.entanglementSpectrum = self.densityMatrixRed.real
            if self.boolEntanglementSpectrum and not self.entanglementSpectrumIndicator:
                tmp = self.onsite * self.systemLevelIterator[0] * self.operatorsRed[0, 0] \
                      + self.interequal * self.operatorsRed[0, 0] * self.operatorsRed[0, 0]
                for i in range(1, len(self.systemLevelIterator)):
                    tmp += self.onsite * self.systemLevelIterator[i] * self.operatorsRed[i, i] \
                           + self.interequal * self.operatorsRed[i, i] * self.operatorsRed[i, i]
                self.entanglementSpectrumEnergy = tmp.diagonal().real

    def updateEntropyRed(self):
        if self.densityMatrixRed is None:
            return
        self.entropyRed = 0
        self.updateEntanglementSpectrum()
        for el in self.entanglementSpectrum:
            if el.real > 0:
                self.entropyRed -= el.real * nplog(el.real)
            elif el.real < -1e-7:
                print('Oh god, there is a negative eigenvalue below 1e-7 ! Namely:', el)

    # end of updateEntropyRed

    def updateOccNumbers(self):
        for m in range(self.m):
            self.occNo[m] = (self.expectValue(self.operators[m, m])).real

    # end of updateOccNumbers

    def updateCorrelations(self):
        for ind in self.correlationsIndices:
            tmpop = self.operators[ind[0], ind[1]]
            tmpnam = self.dataFolder + "correl%i%i" % (ind[0], ind[1])
            for i in range(1, int(len(ind) / 2)):
                tmpop = tmpop.dot(self.operators[ind[i * 2], ind[i * 2 + 1]])
                tmpnam += "%i%i" % (ind[i * 2], ind[i * 2 + 1])
            tmpnam += ".dat"
            tmp = self.expectValue(tmpop)

            tmpfil = open(tmpnam, "a")
            tmpfil.write("%.16e %.16e %.16e\n" % (self.evolTime * self.plotTimeScale, tmp.real, tmp.imag))
            tmpfil.close()

    def updateTotalEnergy(self):
        self.totalEnergy = (self.expectValue(self.hamiltonian)).real

    # end of updateTotalEnergy

    def updateReducedEnergy(self):
        self.reducedEnergy = (self.expectValueRed(self.hamiltonianRed)).real

    # end of updateReducedEnergy

    def evaluateGreen(self, offset=0):
        # offset is given in time as the first time to evaluate
        self.openProgressFile()
        self.filProg.write('Starting evalutation of Green\'s function:\n' + ' 0% ')
        print('Starting evalutation of Green\'s function:\n' + '0% ', end='', flush=True)
        self.closeProgressFile()

        # initialize the evolution operators as the identity (which corresponds to tau=0)
        tmpGreaterEvol = np.identity(self.greenGreaterDim, dtype=self.datType)
        tmpLesserEvol = np.identity(self.greenLesserDim, dtype=self.datType)
        # the time distance (in tau) to sum over is given by the maximum time - the offset time
        time_dist = self.timeSaves[-1] - offset
        # the time steps are (equidistant) given by just the first time step
        dt = self.timeSaves[1] - self.timeSaves[0]
        # the additional dt is needed since the first time element is not counted otherwise
        index_dist_check = int(round((time_dist + dt) / dt))
        # center piece + multiple of two dt distance has to be used since tau comes in steps of 2dt
        if index_dist_check % 2 != 0:
            index_dist_check -= 1
        index_dist = int(index_dist_check)
        # the index of the lowest time coordinate (the offset) - 1
        # (because we start at the middle index_dist is 1 too short)
        offset_index = len(self.timeSaves) - index_dist - 1
        # the actual time offset
        offset_time = self.timeSaves[offset_index]
        # the range to sum up
        bound = int(index_dist / 2)

        # open the green function file for writing
        self.filGreen = open(self.dataFolder + 'green_com%.2f.dat' % (offset_time + bound * dt), 'w')  # t, re, im
        self.filGreen.write('#tau re>_1 im>_1 re<_1 im<_1... COM time is %f\n' % (offset_time + bound * dt))

        # handle the i=0 case => equal time greater is -i*(n_i+1), lesser is -i*n_i
        self.filGreen.write('%.16e ' % 0)
        ind = bound + offset_index
        for m in range(self.m):
            # raising is the higher dimension creation operator, raising.T.c the annihilation
            # this is the greater Green function (without factor of +-i)
            tmpGreenGreater = vdot(self.stateSaves[ind],
                                   self.greenGreaterCreation[m].T.dot(
                                       self.greenGreaterCreation[m].dot(
                                           self.stateSaves[ind]))
                                   )
            # lowering is the lower dimension annihilation operator, lowering.T.c the creation
            # this is the lesser Green function (without factor of +-i)
            tmpGreenLesser = vdot(self.stateSaves[ind],
                                  self.greenLesserAnnihilation[m].T.dot(
                                      self.greenLesserAnnihilation[m].dot(
                                          self.stateSaves[ind]))
                                  )
            # note that the greater Green function is multiplied by -i, which is included in the writing below!
            # note that the lesser Green function is multiplied by -i, which is included in the writing below!
            # first number is real part, second imaginary
            # there is an overall minus sign!
            self.filGreen.write('%.16e %.16e ' % (tmpGreenGreater.imag, -1 * tmpGreenGreater.real))
            self.filGreen.write('%.16e %.16e ' % (tmpGreenLesser.imag, -1 * tmpGreenLesser.real))
        self.filGreen.write(' \n')

        # now start from the first non-zero difference time
        t0 = tm.time()
        t1 = tm.time()
        tavg = 0
        bound_permil = 1000.0 / bound  # use per 1000 to get easier condition for 1% and 10%
        for i in range(1, bound + 1):
            tmpGreaterEvol = tmpGreaterEvol.dot(self.greenGreaterEvolutionMatrix)  # they need to be the squared ones!
            tmpLesserEvol = tmpLesserEvol.dot(self.greenLesserEvolutionMatrix)  # they need to be the squared ones!
            self.filGreen.write('%.16e ' % (2 * dt * i))
            for m in range(self.m):
                # raising is the higher dimension creation operator, raising.T.c the annihilation
                # this is the greater Green function (without factor of +-i)
                tmpGreenGreater = vdot(self.stateSaves[ind - i],
                                       self.greenGreaterCreation[m].T.dot(
                                           tmpGreaterEvol.dot(
                                               self.greenGreaterCreation[m].dot(
                                                   self.stateSaves[ind + i])))
                                       )
                # lowering is the lower dimension annihilation operator, lowering.T.c the creation
                # this is the lesser Green function (without factor of +-i)
                tmpGreenLesser = vdot(self.stateSaves[ind + i],
                                      self.greenLesserAnnihilation[m].T.dot(tmpLesserEvol.dot(
                                          self.greenLesserAnnihilation[m].dot(
                                              self.stateSaves[ind - i])))
                                      )
                # note that the greater Green function is multiplied by -i, which is included in the writing below!
                # note that the lesser Green function is multiplied by -i, which is included in the writing below!
                # first number is real part, second imaginary
                # there is an overall minus sign!
                self.filGreen.write('%.16e %.16e ' % (tmpGreenGreater.imag, -1 * tmpGreenGreater.real))
                self.filGreen.write('%.16e %.16e ' % (tmpGreenLesser.imag, -1 * tmpGreenLesser.real))
            self.filGreen.write(' \n')

            # time estimation start
            if round(i * bound_permil, 1) % 10.0 == 0:
                self.openProgressFile()
                self.filProg.write('.')
                print('.', end='', flush=True)
                if round(i * bound_permil, 1) % 100 == 0:
                    tavg *= int(i - bound / 10)  # calculate from time/step back to unit: time
                    tavg += tm.time() - t1  # add passed time
                    tavg /= i  # average over total number of steps
                    t1 = tm.time()
                    print(' ' + str(int(i * bound_permil / 10)) + "% elapsed: " + time_form(tm.time() - t0),
                          end='', flush=True)

                    self.filProg.write(
                        ' ' + str(int(i * bound_permil / 10)) + "% elapsed: " + time_form(tm.time() - t0))
                    if i != bound:
                        print(" ###### eta: " + time_form(tavg * (bound - i)) + "\n" + str(
                            int(i * bound_permil / 10)) + "% ", end='', flush=True)
                        self.filProg.write(" ###### eta: " + time_form(tavg * (bound - i)) + "\n" + str(
                            int(i * bound_permil / 10)) + "%")
                    else:
                        print('\n')
                        self.filProg.write('\n')
                self.closeProgressFile()
                # time estimation end

        self.filGreen.close()

    def evaluateGreenOnly(self):
        if not self.greenOnly:
            print("WARNING: Green only is not set - several features will not automatically be adjusted.")
        # offset is given in time as the first time to evaluate
        self.openProgressFile()

        # first do the time evolution - negative tau does not affect this, state is evolved to COM time
        self.evolutionMatrix.dot(self.state, out=self.state)
        print("1 - Norm: %.16f \n" % (1.0 - la.norm(self.state)))
        self.filProg.write("1 - Norm: %.16f \n" % (1.0 - la.norm(self.state)))
        self.normalize()
        # for further time evolution it has to be the opposite way - the higher and lower are already conjugate
        self.initEvolutionMatrix(conj=self.boolNegativeTau, greenstep=True)

        # this is the negative time state
        tmpState = self.state
        tmpEvolutionMatrix = self.evolutionMatrix.conjugate().T.copy()

        # initialize the evolution operators as the identity (which corresponds to tau=0)
        tmpGreaterEvol = np.identity(self.greenGreaterDim, dtype=self.datType)
        tmpLesserEvol = np.identity(self.greenLesserDim, dtype=self.datType)

        self.filProg.write('Starting evalutation of Green\'s function:\n' + ' 0% ')
        print('Starting evalutation of Green\'s function:\n' + '0% ', end='', flush=True)
        self.closeProgressFile()

        # open the green function file for writing
        self.filGreen = open(self.dataFolder + 'green_com%.2f.dat' % (int(self.steps * self.deltaT)), 'w')  # t, re, im
        self.filGreen.write('#tau re> im> re< im<, COM time is %f\n' % (self.steps * self.deltaT))

        # if negative tau, set multiplication factor for file writing to -1
        if self.boolNegativeTau:
            tausign = -1
        else:
            tausign = 1

        # initialize the temporary state vectors
        tmpGreaterState_array = np.zeros((self.m, self.dim), dtype=self.datType)
        tmpLesserState_array = np.zeros((self.m, self.dim), dtype=self.datType)

        # handle the i=0 case => equal time greater is -i*(n_i+1), lesser is -i*n_i
        self.filGreen.write('%.16e ' % 0)
        tmpGreaterState_array[0] = self.greenGreaterCreation[0].T.dot(
            self.greenGreaterCreation[0].dot(self.state))
        tmpLesserState_array[0] = self.greenLesserAnnihilation[0].T.dot(
            self.greenLesserAnnihilation[0].dot(self.state))
        for m in range(1, self.m):
            tmpGreaterState_array[m] = self.greenGreaterCreation[m].T.dot(
                self.greenGreaterCreation[m].dot(self.state))
            tmpLesserState_array[m] = self.greenLesserAnnihilation[m].T.dot(
                self.greenLesserAnnihilation[m].dot(self.state))
        # sum over axis no. 0 gives the sum of the individual vectors
        tmpGreenGreater = vdot(self.state, tmpGreaterState_array.sum(axis=0))
        tmpGreenLesser = vdot(self.state, tmpLesserState_array.sum(axis=0))
        # note that the both Green functions are multiplied by -i, which is included in the writing below!
        # first number is real part, second imaginary
        # there is an overall minus sign!
        self.filGreen.write('%.16e %.16e ' % (tmpGreenGreater.imag, -1 * tmpGreenGreater.real))
        self.filGreen.write('%.16e %.16e ' % (tmpGreenLesser.imag, -1 * tmpGreenLesser.real))
        self.filGreen.write(' \n')

        # now start from the first non-zero difference time
        t0 = tm.time()
        t1 = tm.time()
        tavg = 0
        bound_permil = 1000.0 / self.greensteps  # use per 1000 to get easier condition for 1% and 10%
        for i in range(1, self.greensteps + 1):
            # raise the green evolution operator a deltaTau more
            tmpGreaterEvol.dot(self.greenGreaterEvolutionMatrix, out=tmpGreaterEvol)
            tmpLesserEvol.dot(self.greenLesserEvolutionMatrix, out=tmpLesserEvol)
            # evolve the state in time
            self.evolutionMatrix.dot(self.state, out=self.state)
            self.normalize()
            # evolve the tmp state backwards in time
            tmpEvolutionMatrix.dot(tmpState, out=tmpState)
            tmpState /= la.norm(tmpState)

            # the dots only work via contraction with a state vector!
            tmpGreaterState_array[0] = self.greenGreaterCreation[0].T.dot(
                tmpGreaterEvol.dot(
                    self.greenGreaterCreation[0].dot(
                        self.state)))
            tmpLesserState_array[0] = self.greenLesserAnnihilation[0].T.dot(
                tmpLesserEvol.dot(
                    self.greenLesserAnnihilation[0].dot(
                        tmpState)))

            for m in range(1, self.m):
                tmpGreaterState_array[m] = self.greenGreaterCreation[m].T.dot(
                    tmpGreaterEvol.dot(
                        self.greenGreaterCreation[m].dot(
                            self.state)))
                tmpLesserState_array[m] = self.greenLesserAnnihilation[m].T.dot(
                    tmpLesserEvol.dot(
                        self.greenLesserAnnihilation[m].dot(
                            tmpState)))

            tmpGreenGreater = vdot(tmpState, tmpGreaterState_array.sum(axis=0))
            tmpGreenLesser = vdot(self.state, tmpLesserState_array.sum(axis=0))
            self.filGreen.write('%.16e ' % (self.deltaTaugreen * i * tausign))

            # note that the greater Green function is multiplied by -i, which is included in the writing below!
            # note that the lesser Green function is multiplied by -i, which is included in the writing below!
            # first number is real part, second imaginary
            # there is an overall minus sign!
            self.filGreen.write('%.16e %.16e ' % (tmpGreenGreater.imag, -1 * tmpGreenGreater.real))
            self.filGreen.write('%.16e %.16e ' % (tmpGreenLesser.imag, -1 * tmpGreenLesser.real))
            self.filGreen.write(' \n')

            # time estimation start
            if round(i * bound_permil, 1) % 10.0 == 0:
                self.openProgressFile()
                self.filProg.write('.')
                print('.', end='', flush=True)
                if round(i * bound_permil, 1) % 100 == 0:
                    tavg *= int(i - self.greensteps / 10)  # calculate from time/step back to unit: time
                    tavg += tm.time() - t1  # add passed time
                    tavg /= i  # average over total number of steps
                    t1 = tm.time()
                    print(' ' + str(int(i * bound_permil / 10)) + "% elapsed: " + time_form(tm.time() - t0),
                          end='', flush=True)

                    self.filProg.write(
                        ' ' + str(int(i * bound_permil / 10)) + "% elapsed: " + time_form(tm.time() - t0))
                    if i != self.greensteps:
                        print(" ###### eta: " + time_form(tavg * (self.greensteps - i)) + "\n" + str(
                            int(i * bound_permil / 10)) + "% ", end='', flush=True)
                        self.filProg.write(" ###### eta: " + time_form(tavg * (self.greensteps - i)) + "\n" + str(
                            int(i * bound_permil / 10)) + "%")
                    else:
                        print('\n')
                        self.filProg.write('\n')
                self.closeProgressFile()
                # time estimation end

        self.filGreen.close()

    # update everything EXCEPT for total entropy and energy - they are only updated 100 times
    def updateEverything(self):
        self.evolTime += (self.evolStep - self.evolStepTmp) * self.deltaT
        self.evolTime = np.round(self.evolTime, abs(int(np.log10(self.deltaT))))
        self.evolStepTmp = self.evolStep
        self.normalize()
        # only calculate reduced if needed (if just for storing, check if it will be skipped anyways)
        if self.boolReducedEntropy or self.boolReducedEnergy or \
                ((self.boolDMRedStore or self.boolDMRedDiagStore) and
                 (self.dmFileFactor == self.dmFilesSkipFactor or self.dmFileFactor == 0)):
            if self.boolOnlyRed:
                self.reduceDensityMatrixFromState()
            else:
                self.updateDensityMatrix()
                self.reduceDensityMatrix()
            if self.boolDMRedDiagStore:
                self.diagonalizeReducedDensityMatrix()
        elif self.boolDMStore and (self.dmFileFactor == self.dmFilesSkipFactor or self.dmFileFactor == 0) or \
                self.boolTotalEntropy:
            self.updateDensityMatrix()

        if self.boolOccupations:
            self.updateOccNumbers()
        if self.boolReducedEntropy:
            # will automatically calculate entanglement spectrum
            self.updateEntropyRed()
        elif self.boolEntanglementSpectrum:
            # only do this if not already calculated for reduced entropy
            self.updateEntanglementSpectrum()
        if self.boolReducedEnergy:
            self.updateReducedEnergy()
        if self.boolOffDiagOcc:
            self.updateOffDiagOcc()
        if self.boolOffDiagDens:
            self.updateOffDiagDens()
        if self.boolOffDiagDensRed:
            self.updateOffDiagDensRed()
        if len(self.correlationsIndices) != 0:
            self.updateCorrelations()  # note that the file writing is included!

    def timeStep(self):
        self.evolutionMatrix.dot(self.state, out=self.state)

    # end of timeStep

    ###### the magic of time evolution
    def evolve(self):
        # check if state has been normalized yet (or initialized)
        if self.stateNormAbs == 0:
            self.normalize(True)

        if self.boolDataStore:
            self.openFiles()

        self.evolStepTmp = self.evolStep
        stepNo = int(self.dataPoints / 100)
        t0 = t1 = tm.time()  # time before iteration
        self.tavg = 0  # needed for estimation of remaining time
        print('Time evolution\n' + ' 0% ', end='')
        self.openProgressFile()
        self.filProg.write('Time evolution\n' + ' 0% ')
        self.closeProgressFile()
        # percent loop
        for i in range(1, 11):
            # decimal loop
            for ii in range(1, 11):
                self.entanglementSpectrumIndicator = False
                # need only dataPoints steps of size evolStepDist
                for j in range(stepNo):
                    self.updateEverything()  # update always - config file should control what to update
                    if self.boolDataStore:
                        self.writeData()
                    if self.boolRetgreen:
                        self.greenStoreState()

                    # ## Time Step!
                    self.timeStep()
                    self.evolStep += self.evolStepDist

                ######### TMP TMP TMP #########
                # store states for the greens function - temporarily only 100 times
                # if self.boolRetgreen:
                #    self.greenStoreState()

                # calculate total entropy and energy only 100 times, it is time consuming and only a check
                if self.boolTotalEntropy:
                    self.updateEntropy()
                    self.filTotEnt.write('%.16e %.16e \n' % (self.evolTime, self.entropy))

                if self.boolTotalEnergy:
                    self.updateTotalEnergy()
                    self.filTotalEnergy.write('%.16e %.16e \n' % (self.evolTime, self.totalEnergy))

                print('.', end='', flush=True)
                if self.dim > int(1e3) or self.steps > int(1e7):
                    self.openProgressFile()
                    self.filProg.write('.')
                    self.closeProgressFile()

            self.tavg *= int(self.evolStep - self.steps / 10)  # calculate from time/step back to unit: time
            self.tavg += tm.time() - t1  # add passed time
            self.tavg /= self.evolStep  # average over total number of steps
            t1 = tm.time()
            print(' ' + str(i * 10) + "%" + ' 1-norm: ' + str(np.round(1 - self.stateNormAbs, 2)) + ' elapsed: ' +
                  time_form(tm.time() - t0), end='')
            if i != 10:
                print(" ###### eta: " + time_form(self.tavg * (self.steps - self.evolStep)) + "\n" + str(
                    i * 10) + "% ", end='')
            # write to progress log!
            self.openProgressFile()
            self.filProg.write(' ' + str(i * 10) + "%" + ' 1-norm: ' + str(np.round(1 - self.stateNormAbs, 2)) +
                               ' elapsed: ' + time_form(tm.time() - t0))
            if i != 10:
                self.filProg.write(
                    " ###### eta: " + time_form(self.tavg * (self.steps - self.evolStep)) + "\n" + str(
                        i * 10) + "% ")
            else:
                self.filProg.write('\n')
            self.closeProgressFile()

        # so we have datapoints+1 points!
        if self.boolDataStore:
            self.updateEverything()
            self.writeData()
        if self.boolRetgreen:
            self.greenStoreState()

        print('\n' + 'Time evolution finished with average time/datapoint of ' + time_form(
            self.tavg * self.evolStepDist) + '\n', flush=True)
        if self.boolDataStore:
            self.closeFiles()

    # end

    def writeData(self):
        if self.boolDMStore or self.boolDMRedStore or self.boolDMRedDiagStore:
            if self.dmFileFactor == self.dmFilesSkipFactor or self.dmFileFactor == 0:
                self.dmFileFactor = 1
                if not self.boolOnlyRed:
                    if self.boolDMStore:
                        storeMatrix(self.densityMatrix,
                                    self.dataFolder + 'density/densitymatrix_t%u.dat' % self.evolTime)
                    if self.boolDMRedStore:
                        storeMatrix(self.densityMatrixRed,
                                    self.dataFolder + 'red_density/reduced_densitymatrix_t%u.dat' % self.evolTime)
                    if self.boolDMRedDiagStore:
                        np.savetxt(
                            self.dataFolder + 'red_density/reduced_densitymatrix_eigenvalues_t%u.dat' % self.evolTime,
                            self.densityMatrixRedEigenvalues)
                        save_npz(
                            self.dataFolder + 'red_density/reduced_densitymatrix_eigenstates_t%u.npz' % self.evolTime,
                            self.densityMatrixRedEigenvectors)
                    self.dmcount += 1
            else:
                self.dmFileFactor += 1

        if self.boolReducedEnergy:
            self.filRedEnergy.write('%.16e %.16e \n' % (self.evolTime, self.reducedEnergy))

        if self.boolOffDiagOcc:
            self.filOffDiagOcc.write('%.16e ' % self.evolTime)
            for i in range(self.m):
                self.filOffDiagOcc.write('%.16e ' % self.offDiagOcc[i].real)
            self.filOffDiagOcc.write('\n')
            if self.occEnSingle:
                self.filOffDiagOccSingles.write('%.16e ' % self.evolTime)
                for i in range(self.m):
                    for j in range(self.occEnSingle):
                        self.filOffDiagOccSingles.write(
                            '%.16e %.16e ' % (self.offDiagOccSingles[i, j].real, self.offDiagOccSingles[i, j].imag))
                self.filOffDiagOccSingles.write('\n')

        if self.boolReducedEntropy:
            self.filEnt.write('%.16e %.16e \n' % (self.evolTime, self.entropyRed))

        if self.boolEntanglementSpectrum and not self.entanglementSpectrumIndicator:
            self.entanglementSpectrumIndicator = True
            # note that this is the only! time where the time scale is already included
            head = 'n_sys p(n_sys) /// Jt = $f' % (self.evolTime * self.plotTimeScale)
            sort_indices = self.entanglementSpectrumEnergy.argsort()
            np.savetxt(
                self.dataFolder + 'entanglement_spectrum/ent_spec_%.4f.dat' % (self.evolTime * self.plotTimeScale),
                np.column_stack((self.entanglementSpectrumEnergy[sort_indices],
                                 self.entanglementSpectrum[sort_indices])), header=head)

        self.filNorm.write('%.16e %.16e %.16e \n' % (self.evolTime, self.stateNorm, self.stateNormAbs))

        if self.boolOccupations:
            self.filOcc.write('%.16e ' % self.evolTime)
            for m in range(self.m):
                self.filOcc.write('%.16e ' % self.occNo[m])
            self.filOcc.write('\n')

        if self.boolOffDiagDens:
            self.filOffDiagDens.write('%.16e %.16e \n' % (self.evolTime, self.offDiagDens))

        if self.boolOffDiagDensRed:
            self.filOffDiagDensRed.write('%.16e %.16e \n' % (self.evolTime, self.offDiagDensRed))

    def openProgressFile(self):
        self.filProg = open(self.dataFolder + 'progress.log', 'a')

    def closeProgressFile(self):
        self.filProg.close()

    def openFiles(self):
        self.filNorm = open(self.dataFolder + 'norm.dat', 'w')
        if self.boolReducedEntropy:
            self.filEnt = open(self.dataFolder + 'entropy.dat', 'w')
        if self.boolOccupations:
            self.filOcc = open(self.dataFolder + 'occupation.dat', 'w')
        if self.boolTotalEntropy:
            self.filTotEnt = open(self.dataFolder + 'total_entropy.dat', 'w')
        if self.boolTotalEnergy:
            self.filTotalEnergy = open(self.dataFolder + 'total_energy.dat', 'w')
        if self.boolReducedEnergy:
            self.filRedEnergy = open(self.dataFolder + 'reduced_energy.dat', 'w')
        if self.boolOffDiagOcc:
            self.filOffDiagOcc = open(self.dataFolder + 'offdiagocc.dat', 'w')
            if self.occEnSingle:
                self.filOffDiagOccSingles = open(self.dataFolder + 'offdiagoccsingle.dat', 'w')
        if self.boolOffDiagDens:
            self.filOffDiagDens = open(self.dataFolder + 'offdiagdens.dat', 'w')
        if self.boolOffDiagDensRed:
            self.filOffDiagDensRed = open(self.dataFolder + 'offdiagdensred.dat', 'w')

    # close all files
    def closeFiles(self):
        self.filNorm.close()
        if self.boolReducedEntropy:
            self.filEnt.close()
        if self.boolOccupations:
            self.filOcc.close()
        if self.boolTotalEntropy:
            self.filTotEnt.close()
        if self.boolTotalEnergy:
            self.filTotalEnergy.close()
        if self.boolReducedEnergy:
            self.filRedEnergy.close()
        if self.boolOffDiagOcc:
            self.filOffDiagOcc.close()
            if self.occEnSingle:
                self.filOffDiagOccSingles.close()
        if self.boolOffDiagDens:
            self.filOffDiagDens.close()
        if self.boolOffDiagDensRed:
            self.filOffDiagDensRed.close()

    def plotDMAnimation(self, stepSize=1):
        ep.plotDensityMatrixAnimation(self.steps, self.deltaT, self.dmFiles, stepSize, framerate=self.dmFilesFPS)

    def plotDMRedAnimation(self, stepSize=1):
        ep.plotDensityMatrixAnimation(self.steps, self.deltaT, self.dmFiles, stepSize, 1, framerate=self.dmFilesFPS)

    def plotData(self):
        ep.plotData(self)

    @staticmethod
    def plotHamiltonian():
        ep.plotHamiltonian()

    def plotOccEnbasis(self):
        ep.plotOccs(self)

    def plotOffDiagOccSingles(self):
        ep.plotOffDiagOccSingles(self)

    def plotTimescale(self):
        ep.plotTimescale(self)

    def plot(self):
        if self.boolPlotData:
            self.plotData()
        if self.boolPlotHamiltonian:
            self.plotHamiltonian()
        if self.boolPlotDMAnimation:
            self.plotDMAnimation(self.dmFilesStepSize)
        if self.boolPlotDMRedAnimation:
            self.plotDMRedAnimation(self.dmFilesStepSize)
        if self.boolPlotOccEn:
            self.plotOccEnbasis()
        if self.boolPlotTimescale:
            self.plotTimescale()
        if self.boolPlotOffDiagOccSingles:
            self.plotOffDiagOccSingles()
        if self.boolClear:
            prepFolders(True, folderpath=self.dataFolder)  # clear out all the density matrix folders

    @staticmethod
    def clearDensityData(path="./data/"):
        prepFolders(True, path)


def prepFolders(clearbool=0, densbool=0, reddensbool=0, spectralbool=0, entspecbool=0, cleanbeforebool=0,
                folderpath="./data/"):
    # create the needed folders
    if cleanbeforebool:
        if os.path.exists(folderpath):
            for root, dirs, files in os.walk(folderpath, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            print("Old data folder has been completely cleared")
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    if densbool:
        if not os.path.exists(folderpath + "density/"):
            os.mkdir(folderpath + "density/")
    if reddensbool:
        if not os.path.exists(folderpath + "red_density/"):
            os.mkdir(folderpath + "red_density/")
    if spectralbool:
        if not os.path.exists(folderpath + "spectral/"):
            os.mkdir(folderpath + "spectral/")
    if entspecbool:
        if not os.path.exists(folderpath + "entanglement_spectrum/"):
            os.mkdir(folderpath + "entanglement_spectrum/")
    if not os.path.exists(folderpath + "plots/"):
        os.mkdir(folderpath + "plots/")
    # remove the old stuff
    if clearbool:
        if os.path.isfile(folderpath + "density/densmat0.dat"):
            for root, dirs, files in os.walk(folderpath + 'density/', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            print("Cleared density folder")

        if os.path.isfile(folderpath + "red_density/densmat0.dat"):
            for root, dirs, files in os.walk(folderpath + 'red_density/', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            print("Cleared reduced density folder")


# calculate the number of Coefficients
def dimOfBasis(N, m):
    return int(binom(N + m - 1, N))


# fill the fock basis, offset says where to start
def fillBasis(basis, N, m, offset=0):
    if m != 1:
        counter = [offset]
        for n in range(m - 1):
            counter[0] = offset
            a(N, m, n, basis, N, m, counter)
        counter[0] = offset
        am(N, m, m - 1, basis, N, m, counter)
    else:
        basis[offset, 0] = int(N)
        # end


# https://en.wikipedia.org/wiki/Skew_normal_distribution
def gaussian(x, mu, sigm=1, norm=1, skw=0):
    tmp = np.exp(-(x - mu) ** 2 / (2 * sigm ** 2))
    if norm:
        tmp /= np.sqrt(2 * np.pi * sigm ** 2)
    if skw != 0:
        tmp *= (1 + erf(np.array([skw * (x - mu) / (sigm * sqrt(2))]))[0])
    return tmp


# mu=(a+b)/2 and sigma^2 = (b-a)^2 / 12 so a=mu-sqrt(3)*sigma and b=mu-sqrt(3)*sigma
def rect(x, mu, sigm=1, norm=1):
    if np.abs(x - mu) <= sqrt(3) * sigm:
        tmp = 1
    else:
        tmp = 0

    if norm:
        tmp /= 2 * sqrt(3) * sigm
    return tmp


# gives offsets for reduced density matrix
def basisOffsets(N, m):
    offsets = np.zeros((N + 2), dtype=np.int32)
    # set starting positions (counting from N,m-1 downwards) => results in rdm being block matrix of decreasing N_sub
    # set first one for avoiding exception in first nn value
    offsets[N] = 0
    # then following is offset[N-1] = offset[N] + #el(N,m-1) / however note offset[N]=0
    # offset[N-2] = offset[N-1] + #el(N-1,m_red)
    for i in reversed(range(-1, N)):
        offsets[i] = offsets[i + 1] + dimOfBasis(i + 1, m)
    # note: offsets[N+1] = dim of basis
    return offsets


def fillReducedBasis(basis, N, m, offsets):
    for i in range(N + 1):
        fillBasis(basis, i, m, offsets[i])
        # end


# filling arrays for l != m-1
def a(N, m, l, basis, Nsys, msys, counter):
    if m == msys - l:
        for n in range(N + 1):
            nn = 0
            while nn < dimOfBasis(n, m - 1):
                basis[counter[0]][l] = int(N - n)
                counter[0] += 1
                nn += 1
    else:
        for n in reversed(range(N + 1)):
            a(N - n, m - 1, l, basis, Nsys, msys, counter)
            # end


# filling arrays for l == m-1 (order is other way round)
def am(N, m, l, basis, Nsys, msys, counter):
    if m == msys:
        am(N, m - 1, l, basis, Nsys, msys, counter)
    elif m == msys - l:
        for n in reversed(range(N + 1)):
            basis[counter[0]][l] = int(N - n)
            counter[0] += 1
    else:
        for n in reversed(range(N + 1)):
            am(N - n, m - 1, l, basis, Nsys, msys, counter)
            # end


def basis2dict(basis, dim):
    # create an empty dictionary with states in occupation number repres. as tuples being the keys
    tup = tuple(tuple(el) for el in basis)
    dRet = dict.fromkeys(tup)
    # for correct correspondence go through the basis tuple and put in the vector-number corresponding to the given tuple
    for i in range(dim):
        dRet[tup[i]] = i
    return dRet


# note that all the elements here are sparse matrices! one has to use .toarray() to get them done correctly
def quadraticArray(sysVar):
    retArr = np.empty((sysVar.m, sysVar.m), dtype=csr_matrix)
    # off diagonal
    for i in range(sysVar.m):
        for j in range(i):
            retArr[i, j] = getQuadratic(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(sysVar.m):
        retArr[i, i] = getQuadratic(sysVar, i, i)
    return retArr


# note that all the elements here are sparse matrices! one has to use .toarray() to get them done correctly
# please also note the index shift - from low to high but without the traced out, e.g.
# m=4 trace out 1,2 -> 0,1 of reduced array corresponds to level 0 and 4 of whole system
def quadraticArrayRed(sysVar):
    retArr = np.empty((sysVar.mRed, sysVar.mRed), dtype=csr_matrix)
    # off diagonal
    for i in range(sysVar.mRed):
        for j in range(i):
            retArr[i, j] = getQuadraticRed(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(sysVar.mRed):
        retArr[i, i] = getQuadraticRed(sysVar, i, i)
    return retArr


def quadraticArrayGreenLesser(sysVar):
    retArr = np.empty((sysVar.m, sysVar.m), dtype=csr_matrix)
    # off diagonal
    for i in range(sysVar.m):
        for j in range(i):
            retArr[i, j] = getQuadraticGreenLesser(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(sysVar.m):
        retArr[i, i] = getQuadraticGreenLesser(sysVar, i, i)
    return retArr


def quadraticArrayGreenGreater(sysVar):
    retArr = np.empty((sysVar.m, sysVar.m), dtype=csr_matrix)
    # off diagonal
    for i in range(sysVar.m):
        for j in range(i):
            retArr[i, j] = getQuadraticGreenGreater(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(sysVar.m):
        retArr[i, i] = getQuadraticGreenGreater(sysVar, i, i)
    return retArr


# quadratic term in 2nd quantization for transition from m to l
# matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadratic(sysVar, l, m):
    entries = sysVar.dim - dimOfBasis(sysVar.N, sysVar.m - 1)
    data = np.zeros(entries, dtype=np.float64)
    row = np.zeros(entries, dtype=np.float64)
    col = np.zeros(entries, dtype=np.float64)
    tmp = np.zeros(sysVar.m, dtype=np.int)
    counter = 0
    for el in sysVar.basis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row[counter] = sysVar.basisDict[tuple(tmp)]
            col[counter] = sysVar.basisDict[tuple(el)]
            data[counter] = np.float64(sqrt(el[m]) * sqrt(tmp[l]))
            counter += 1

    retmat = coo_matrix((data, (row, col)), shape=(sysVar.dim, sysVar.dim), dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


# quadratic term in 2nd quantization for transition from m to l
# matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadraticRed(sysVar, l, m):
    entries = sysVar.dimRed - dimOfBasis(sysVar.N, sysVar.mRed - 1)
    data = np.zeros(entries, dtype=np.float64)
    row = np.zeros(entries, dtype=np.float64)
    col = np.zeros(entries, dtype=np.float64)
    tmp = np.zeros(sysVar.mRed, dtype=np.int)
    counter = 0
    for el in sysVar.basisRed:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row[counter] = sysVar.basisDictRed[tuple(tmp)]
            col[counter] = sysVar.basisDictRed[tuple(el)]
            data[counter] = np.float64(sqrt(el[m]) * sqrt(tmp[l]))
            counter += 1

    retmat = coo_matrix((data, (row, col)), shape=(sysVar.dimRed, sysVar.dimRed), dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


# quadratic term in 2nd quantization for transition from m to l for system with one particle removed
# matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadraticGreenLesser(sysVar, l, m):
    entries = sysVar.greenLesserDim - dimOfBasis(sysVar.N - 1, sysVar.m - 1)
    data = np.zeros(entries, dtype=np.float64)
    row = np.zeros(entries, dtype=np.float64)
    col = np.zeros(entries, dtype=np.float64)
    tmp = np.zeros(sysVar.m, dtype=np.int)
    counter = 0
    for el in sysVar.greenLesserBasis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row[counter] = sysVar.greenLesserBasisDict[tuple(tmp)]
            col[counter] = sysVar.greenLesserBasisDict[tuple(el)]
            data[counter] = np.float64(sqrt(el[m]) * sqrt(tmp[l]))
            counter += 1

    retmat = coo_matrix((data, (row, col)), shape=(sysVar.greenLesserDim, sysVar.greenLesserDim),
                        dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


# quadratic term in 2nd quantization for transition from m to l for system with one particle added
# matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadraticGreenGreater(sysVar, l, m):
    entries = sysVar.greenGreaterDim - dimOfBasis(sysVar.N + 1, sysVar.m - 1)
    data = np.zeros(entries, dtype=np.float64)
    row = np.zeros(entries, dtype=np.float64)
    col = np.zeros(entries, dtype=np.float64)
    tmp = np.zeros(sysVar.m, dtype=np.int)
    counter = 0
    for el in sysVar.greenGreaterBasis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row[counter] = sysVar.greenGreaterBasisDict[tuple(tmp)]
            col[counter] = sysVar.greenGreaterBasisDict[tuple(el)]
            data[counter] = np.float64(sqrt(el[m]) * sqrt(tmp[l]))
            counter += 1

    retmat = coo_matrix((data, (row, col)), shape=(sysVar.greenGreaterDim, sysVar.greenGreaterDim),
                        dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


# array elements are NO matrix! just numpy array!
# This will take very long to create and use up much memory, please consider doing it on the fly only for needed elements.           
def quarticArray(sysVar):
    retArr = np.empty((sysVar.m, sysVar.m, sysVar.m, sysVar.m), dtype=csr_matrix)
    # TODO: use transpose property
    for k in range(sysVar.m):
        for l in range(sysVar.m):
            for m in range(sysVar.m):
                for n in range(sysVar.m):
                    retArr[k, l, m, n] = getQuartic(sysVar, k, l, m, n)
    return retArr


# array elements are NO matrix! just numpy array!
# This will take very long to create and use up much memory, please consider doing it on the fly only for needed elements.           
def quarticArrayRed(sysVar):
    retArr = np.empty((sysVar.mRed, sysVar.mRed, sysVar.mRed, sysVar.mRed), dtype=csr_matrix)
    # TODO: use transpose property
    for k in range(sysVar.mRed):
        for l in range(sysVar.mRed):
            for m in range(sysVar.mRed):
                for n in range(sysVar.mRed):
                    retArr[k, l, m, n] = getQuarticRed(sysVar, k, l, m, n)
    return retArr


def getQuartic(sysVar, k, l, m, n):
    if l != m:
        return (sysVar.operators[k, m].dot(sysVar.operators[l, n])).copy()
    else:
        return ((sysVar.operators[k, m].dot(sysVar.operators[l, n])) - sysVar.operators[k, n]).copy()


def getQuarticRed(sysVar, k, l, m, n):
    if l != m:
        return (getQuadraticRed(sysVar, k, m).dot(getQuadraticRed(sysVar, l, n))).copy()
    else:
        return (
                (getQuadraticRed(sysVar, k, m).dot(getQuadraticRed(sysVar, l, n))) - getQuadraticRed(sysVar, k,
                                                                                                     n)).copy()


def getQuarticGreen(quadops, k, l, m, n):
    if l != m:
        return (quadops[k, m].dot(quadops[l, n])).copy()
    else:
        return ((quadops[k, m].dot(quadops[l, n])) - quadops[k, n]).copy()


# destruction operator (N -> N-1)
# adjoint of this is creation on N-1
def getLesserAnnihilation(sysVar, l):
    entries = sysVar.dim - dimOfBasis(sysVar.N, sysVar.m - 1)
    data = np.zeros(entries, dtype=np.float64)
    row = np.zeros(entries, dtype=np.float64)
    col = np.zeros(entries, dtype=np.float64)
    tmp = np.zeros(sysVar.m, dtype=np.int)
    counter = 0
    for el in sysVar.basis:
        if el[l] != 0:
            tmp = el.copy()
            tmp[l] -= 1
            row[counter] = sysVar.greenLesserBasisDict[tuple(tmp)]
            col[counter] = sysVar.basisDict[tuple(el)]
            data[counter] = np.sqrt(el[l], dtype=np.float64)
            counter += 1

    retmat = coo_matrix((data, (row, col)), shape=(sysVar.greenLesserDim, sysVar.dim), dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


# creation operator (N -> N+1)
# adjoint of this is annihilation on N+1
def getGreaterCreation(sysVar, l):
    entries = sysVar.dim
    data = np.zeros(entries, dtype=np.float64)
    row = np.zeros(entries, dtype=np.float64)
    col = np.zeros(entries, dtype=np.float64)
    tmp = np.zeros(sysVar.m, dtype=np.int)
    counter = 0
    for el in sysVar.basis:
        tmp = el.copy()
        tmp[l] += 1
        row[counter] = sysVar.greenGreaterBasisDict[tuple(tmp)]
        col[counter] = sysVar.basisDict[tuple(el)]
        data[counter] = np.sqrt(tmp[l], dtype=np.float64)
        counter += 1

    retmat = coo_matrix((data, (row, col)), shape=(sysVar.greenGreaterDim, sysVar.dim), dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


def time_elapsed(t0, divider, decimals=0):
    t_el = tm.time() - t0
    if divider == 60:
        t_min = t_el // 60
        t_sec = t_el % 60
        return "%2im %2is" % (int(t_min), int(t_sec))
    else:
        t_sec = t_el / divider
        return str(round(t_sec, decimals)) + "s"


def time_form(seconds):
    magnitude = int(np.log10(seconds))
    unit = ''
    divider = 1
    if magnitude < 0:
        if magnitude >= -3:
            unit = 'ms'
            divider = 1e-3
        elif magnitude >= -6:
            unit = chr(956) + 's'
            divider = 1e-6
        elif magnitude >= -9:
            unit = 'ns'
            divider = 1e-9
        elif magnitude >= -12:
            unit = 'ps'
            divider = 1e-12
    else:
        if seconds < 1e2:  # 100 seconds
            unit = 's'
            divider = 1
        elif seconds < 6e3:  # 100 mins
            unit = 'm'
            divider = 60
        elif seconds < 3.569e6:  # 999 hours
            unit = 'h'
            divider = 3.6e3
        else:
            unit = 'd'
            divider = 8.64e4
    return str(int(seconds / divider)) + unit


# stores the matrix of dimension sysvar[2] in a file
def storeMatrix(mat, fil, absOnly=0, stre=True, stim=True, stabs=True):
    matDim = mat.shape[0]
    if absOnly == 0:
        # assume dot + 3 letter ending e.g. .dat
        fname = fil[:-4]
        fend = fil[-4:]
        if stabs:
            f = open(fil, 'w')
        if stim:
            fimag = open(fname + '_im' + fend, 'w')
        if stre:
            freal = open(fname + '_re' + fend, 'w')
        for n in range(matDim):
            for nn in range(matDim - 1):
                if stabs:
                    f.write('%.16e ' % np.abs(mat[(n, nn)]))
                if stim:
                    fimag.write('%.16e ' % (mat[(n, nn)]).imag)
                if stre:
                    freal.write('%.16e ' % (mat[(n, nn)]).real)
            if stabs:
                f.write('%.16e\n' % np.abs(mat[(n, matDim - 1)]))
            if stim:
                fimag.write('%.16e\n' % (mat[(n, matDim - 1)]).imag)
            if stre:
                freal.write('%.16e\n' % (mat[(n, matDim - 1)]).real)
        if stabs:
            f.close()
        if stim:
            fimag.close()
        if stre:
            freal.close()
    else:
        f = open(fil, 'w')
        # assume dot + 3 letter ending e.g. .dat
        for n in range(matDim):
            for nn in range(matDim - 1):
                f.write('%.16e ' % np.abs(mat[(n, nn)]))
            f.write('%.16e\n' % np.abs(mat[(n, matDim - 1)]))
        f.close()


multiprocDic = {}


def initWorker(state, stateConj, densmat, dim, dimRed, datType, constructorRaw, constructorRawConj):
    multiprocDic['state'] = state
    multiprocDic['stateConj'] = stateConj
    multiprocDic['densityMatrix'] = densmat
    multiprocDic['shapeState'] = (dim,)
    multiprocDic['shapeMat'] = (dim, dim)
    multiprocDic['shapeRed'] = (dimRed, dimRed)
    multiprocDic['datType'] = datType
    for i in range(len(constructorRaw)):
        multiprocDic['retArray%i' % i] = constructorRaw['%i' % i]
        multiprocDic['retArrayConj%i' % i] = constructorRawConj['%i' % i]


def reduceDensityMatrixFromState_mp_helper_small(iterator_array, id):
    state_np = \
        np.frombuffer(multiprocDic['state'], dtype=multiprocDic['datType']).reshape(multiprocDic['shapeState'])
    stateConj_np = \
        np.frombuffer(multiprocDic['stateConj'], dtype=multiprocDic['datType']).reshape(multiprocDic['shapeState'])
    densMat_constructor = \
        np.frombuffer(multiprocDic['retArray%i' % id], dtype=multiprocDic['datType']).reshape(len(iterator_array))
    i = 0
    for el in iterator_array:
        densMat_constructor[0] = state_np[el[2]] * stateConj_np[el[3]]
        i += 1
    return 1


def reduceDensityMatrixFromState_mp_helper(iterator_array, id):
    state_np = \
        np.frombuffer(multiprocDic['state'], dtype=multiprocDic['datType']).reshape(multiprocDic['shapeState'])
    stateConj_np = \
        np.frombuffer(multiprocDic['stateConj'], dtype=multiprocDic['datType']).reshape(multiprocDic['shapeState'])
    densMat_constructor = \
        np.frombuffer(multiprocDic['retArray%i' % id], dtype=multiprocDic['datType']).reshape(len(iterator_array))
    densMat_constructor_conjugate = \
        np.frombuffer(multiprocDic['retArrayConj%i' % id], dtype=multiprocDic['datType']).reshape(len(iterator_array))

    i = 0
    for el in iterator_array:
        densMat_constructor[i] = state_np[el[2]] * stateConj_np[el[3]]
        if el[0] != el[1]:
            densMat_constructor_conjugate[i] = state_np[el[3]] * stateConj_np[el[2]]
        else:
            densMat_constructor_conjugate[i] = 0
        i += 1
    return 1


def reduceDensityMatrix_mp_helper_small(iterator_array, id):
    densMat_np = \
        np.frombuffer(multiprocDic['densityMatrix'], dtype=multiprocDic['datType']).reshape(multiprocDic['shapeMat'])
    densMat_constructor = \
        np.frombuffer(multiprocDic['retArray%i' % id], dtype=multiprocDic['datType']).reshape(len(iterator_array))

    i = 0
    for el in iterator_array:
        densMat_constructor[i] = densMat_np[el[2], el[3]]
        i += 1
    return 1


def reduceDensityMatrix_mp_helper(iterator_array, id):
    densMat_np = \
        np.frombuffer(multiprocDic['densityMatrix'], dtype=multiprocDic['datType']).reshape(multiprocDic['shapeMat'])
    densMat_constructor = \
        np.frombuffer(multiprocDic['retArray%i' % id], dtype=multiprocDic['datType']).reshape(len(iterator_array))
    densMat_constructor_conjugate = \
        np.frombuffer(multiprocDic['retArrayConj%i' % id], dtype=multiprocDic['datType']).reshape(len(iterator_array))

    i = 0
    for el in iterator_array:
        densMat_constructor[i] = densMat_np[el[2], el[3]]
        if el[0] != el[1]:
            densMat_constructor_conjugate[i] = densMat_np[el[3], el[2]]
        else:
            densMat_constructor_conjugate[i] = 0
        i += 1
    return 1
