# This program is explicitly written for PYTHON3.X and will not work under Python2.X

import numpy as np
from numpy import dot as npdot
import shutil
from scipy.special import binom
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import identity as spidentity
from scipy.special import erf
import scipy.linalg as la
from numpy import einsum as npeinsum
import time as tm
import os, configparser
import entPlot as ep
from numpy import sqrt
from numpy import log as nplog
from numpy.linalg import matrix_power as npmatrix_power
from configparser import ConfigParser
from scipy.special.basic import factorial

# manyparticle system class
class mpSystem:
    # N = total particle number, m = number of states in total, redStates = array of state indices to be traced out
    def __init__(self, cFile="default.ini", dtType=np.complex128, plotOnly=False):
        self.confFile = cFile
        prepFolders(0)
        self.loadConfig()
        # mask selects only not traced out states
        self.mask = np.ones((self.m) , dtype=bool)
        for k in self.kRed:
            self.mask[k] = False
        
        self.dim = dimOfBasis(self.N, self.m)  # dimension of basis
        ###### system variables
        if not plotOnly:
            self.datType = dtType
            self.basis = np.zeros((self.dim , self.m) , dtype=np.int)
            fillBasis(self.basis, self.N, self.m)
            self.basisDict = basis2dict(self.basis, self.dim)
            # note that there is an additional dimension there! needed for fast multiplication algorithm
            self.state = np.zeros((self.dim, 1) , dtype=self.datType)
            # parameter for storing in file
            self.stateNorm = 0
            self.stateNormAbs = 0
            self.stateNormCheck = 1e1  # check if norm has been supressed too much
            self.densityMatrix = []  # do not initialize yet - it wait until hamiltonian decomposition has been done for memory efficiency 
            self.densityMatrixInd = False
            self.entropy = 0
            self.energy = 0
            self.operators = quadraticArray(self)
            self.occNo = np.zeros(self.m, dtype=np.float64)
            # hamiltonian - initialized with zeros (note - datatype is not! complex)
            self.hamiltonian = coo_matrix(np.zeros((self.dim, self.dim)), shape=(self.dim, self.dim), dtype=np.float64).tocsr()
            # matrix for time evolution - initially empty
            self.evolutionMatrix = None
            # eigenvalue and vectors
            self.eigVals = []
            self.eigVects = []
            self.eigInd = False
            # iteration step
            self.evolStep = 0
            self.evolStepTmp = 0
            self.evolTime = 0
            self.tavg = 0  # needed for estimation of remaining time
            self.dmcount = 0  # needed for numbering of density matrix files
            ###### variables for the partial trace algorithm
            self.mRed = self.m - len(self.kRed)
            self.mRedComp = len(self.kRed)
            self.entropyRed = 0
    
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
                self.dimRed = dimOfBasis(self.N , (self.mRed + 1))
                self.offsetsRed = basisOffsets(self.N, self.mRed)
                self.basisRed = np.zeros((self.dimRed , self.mRed) , dtype=np.int)
                fillReducedBasis(self.basisRed, self.N, self.mRed, self.offsetsRed)
                self.basisDictRed = basis2dict(self.basisRed, self.dimRed) #only!! neded for reduced space operators
                self.dimRedComp = dimOfBasis (self.N , (self.mRedComp + 1))
                self.offsetsRedComp = basisOffsets(self.N, self.mRedComp)
                self.basisRedComp = np.zeros((self.dimRedComp , self.mRedComp) , dtype=np.int)
                fillReducedBasis(self.basisRedComp, self.N, self.mRedComp, self.offsetsRedComp)
                self.densityMatrixRed = np.zeros((self.dimRed , self.dimRed) , dtype=self.datType)
                self.iteratorRed = np.zeros((0 , 4), dtype=np.int32)
                self.initIteratorRed() 
            
            ### Spectral
            if self.boolSpectral or self.boolRetgreen:
                ## lo
                self.specLoDim = dimOfBasis(self.N - 1, self.m)
                self.specLoBasis = np.zeros((self.specLoDim , self.m) , dtype=np.int)
                fillBasis(self.specLoBasis, self.N-1, self.m)
                self.specLoBasisDict = basis2dict(self.specLoBasis, self.specLoDim)
                self.specLoHamiltonian = coo_matrix(np.zeros((self.specLoDim, self.specLoDim)), shape=(self.specLoDim, self.specLoDim), dtype=np.float64).tocsr()
                ## hi
                self.specHiDim = dimOfBasis(self.N + 1, self.m)
                self.specHiBasis = np.zeros((self.specHiDim , self.m) , dtype=np.int)
                fillBasis(self.specHiBasis, self.N+1, self.m)
                self.specHiBasisDict = basis2dict(self.specHiBasis, self.specHiDim)
                self.specHiHamiltonian = coo_matrix(np.zeros((self.specHiDim, self.specHiDim)), shape=(self.specHiDim, self.specHiDim), dtype=np.float64).tocsr()
                if self.boolRetgreen:
                    self.green = np.zeros(self.m, dtype = self.datType)
                    self.stateSaves = [] #append with time dep. state vector
                    self.timeSaves = [] #append with time of saved state vector
                    self.specLoEvolutionMatrix = None
                    self.specHiEvolutionMatrix = None
                    self.specLowering = []
                    self.specRaising = []
                    #fill'em
                    for i in range(0,self.m):
                        #note that the lowering operator transposed is the raising op. of the lower dimension space
                        self.specLowering.append(getLoweringSpec(self, i))
                        # the raising operator transposed is the lowering op. of the higher dimension space
                        self.specRaising.append(getRaisingSpec(self, i))
                        
                if self.boolSpectral:
                    ## Lo
                    # energy basis single operator matrix (k,i)
                    self.specLoMatrix = []
                    self.specLoEigVals = []
                    self.specLoEigVects = []
                    self.specLoEigInd = False
                    self.specLoInd = False
                    ## hi
                    # energy basis single operator matrix (k,i)
                    self.specHiMatrix = []
                    self.specHiEigVals = []
                    self.specHiEigVects = []
                    self.specHiEigInd = False
                    self.specHiInd = False
    # end of init
    
    ###### reading from config file
    def loadConfig(self):
        configParser = configparser.RawConfigParser()
        #read the defaults
        configParser.read('./default.ini')  
        #read the actual config file
        configParser.read('./' + self.confFile)
        # ## system parameters
        self.N = int(configParser.getfloat('system', 'N'))
        self.m = int(configParser.getfloat('system', 'm'))
        self.kRed = configParser.get('system', 'kred').split(',')
        if len(self.kRed[0]) == 0:
            self.kRed = []
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

        # ## file management
        self.dataPoints = int(configParser.getfloat('filemanagement', 'datapoints'))
        self.dmFilesSkipFactor = int(configParser.getfloat('filemanagement', 'dmfile_skipfactor'))
        self.boolClear = configParser.getboolean('filemanagement', 'clear')
        self.boolDataStore = configParser.getboolean('filemanagement', 'datastore')
        self.boolDMStore = configParser.getboolean('filemanagement', 'dmstore') 
        self.boolDMRedStore = configParser.getboolean('filemanagement', 'dmredstore') 
        self.boolHamilStore = configParser.getboolean('filemanagement', 'hamilstore')
        self.boolOccEnStore = configParser.getboolean('filemanagement', 'occenstore')
        self.boolEngyStore = configParser.getboolean('filemanagement', 'energiesstore')
        self.boolDecompStore = configParser.getboolean('filemanagement', 'decompstore')
        self.boolDiagExpStore = configParser.getboolean('filemanagement', 'diagexpstore')
        self.boolSpectral = configParser.getboolean('filemanagement', 'spectral')
        self.boolRetgreen = configParser.getboolean('filemanagement', 'retgreen')
        # ## calculation-parameters
        self.boolOnlyRed = configParser.getboolean('calcparams', 'onlyreduced')
        self.boolTotalEnt = configParser.getboolean('calcparams', 'totalentropy')
        self.boolTotalEnergy = configParser.getboolean('calcparams', 'totalenergy')
        # ## plotting booleans and parameters
        self.boolPlotData = configParser.getboolean('plotbools', 'data')
        self.boolPlotAverages = configParser.getboolean('plotbools', 'averages')
        self.boolPlotHamiltonian = configParser.getboolean('plotbools', 'hamiltonian')
        self.boolPlotDMAnimation = configParser.getboolean('plotbools', 'densistymatrix')
        self.boolPlotDMRedAnimation = configParser.getboolean('plotbools', 'reducedmatrix')
        self.boolPlotOccEn = configParser.getboolean('plotbools', 'occen')
        self.boolPlotEngy = configParser.getboolean('plotbools', 'energies')
        self.boolPlotDecomp = configParser.getboolean('plotbools', 'decomposition')
        self.boolPlotDiagExp = configParser.getboolean('plotbools', 'diagexp')
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
        self.plotLoAvgPerc = configParser.getfloat('plotvals', 'loavgperc')/100.0
        # normally some coefficient in the hamiltonian (J or t)
        self.plotTimeScale = configParser.getfloat('plotvals', 'timescale')
        
        self.evolStepDist = int(self.steps / self.dataPoints)
        if self.evolStepDist < 100:
            self.steps = 100 * self.dataPoints
            self.evolStepDist = 100
            print('Number of steps must be at least factor 100 larger than datapoints! New number of steps: %e' % self.steps)
        self.dmFiles = self.dataPoints / self.dmFilesSkipFactor    
        
        if self.dataPoints > self.steps:
            self.dataPoints = self.steps / 100
            print('Number of data points was larger than number of steps - think again! Fixed the number of data points to be: %e' % self.dataPoints)
        
    ###### Methods:
    def updateDensityMatrix(self):
        if self.densityMatrixInd == False:
            # there might be a memory reallocation error with np.outer... however, initialization is always nice
            self.densityMatrix = np.zeros((self.dim , self.dim) , dtype=self.datType)
            self.densMatrixInd = True
            
        self.densityMatrix = np.outer(self.state[:, 0], np.conjugate(self.state[:, 0]))
    # end of updateDensityMatrix
    
    def initIteratorRed(self):
        el1 = np.zeros((self.m) , dtype=np.int)
        el2 = np.zeros((self.m) , dtype=np.int)
        for i in reversed(range(0, self.N + 1)):
            for j in range(self.offsetsRed[i], self.offsetsRed[i - 1]):
                for jj in range(j, self.offsetsRed[i - 1]):
                    for k in range(self.offsetsRedComp[self.N - i], self.offsetsRedComp[self.N - i - 1]):
                        el1[self.mask] = self.basisRed[j]
                        el1[~self.mask] = self.basisRedComp[k]
                        el2[self.mask] = self.basisRed[jj]
                        el2[~self.mask] = self.basisRedComp[k]
                        self.iteratorRed = np.append(self.iteratorRed, [[j, jj, self.basisDict[tuple(el1)], self.basisDict[tuple(el2)]]], axis=0)
    # end of initTest
    
    def reduceDensityMatrix(self):
        if self.densityMatrixRed is None:
            return
        self.densityMatrixRed.fill(0)
        for el in self.iteratorRed:
            self.densityMatrixRed[el[0], el[1]] += self.densityMatrix[el[2], el[3]]
            if el[0] != el[1]:
                self.densityMatrixRed[el[1], el[0]] += self.densityMatrix[el[3], el[2]]
    
    def reduceDensityMatrixFromState(self):
        stTmp = np.conjugate(self.state)
        if self.densityMatrixRed is None:
            return
        self.densityMatrixRed.fill(0)
        for el in self.iteratorRed:
            self.densityMatrixRed[el[0], el[1]] += self.densityMatrix[el[2], el[3]]
            if el[0] != el[1]:
                self.densityMatrixRed[el[1], el[0]] += self.state[el[3], 0] * stTmp[el[2], 0]
    # end of reduceDensityMatrixFromState

    def reduceMatrix(self,matrx):
        tmpret = np.zeros((self.dimRed,self.dimRed))
        for el in self.iteratorRed:
            tmpret[el[0], el[1]] += matrx[el[2], el[3]]
            if el[0] != el[1]:
                tmpret[el[1], el[0]] += matrx[el[3], el[2]]
        return tmpret
    
    # hamiltonian with equal index interaction different to non equal index interaction
    def initHamiltonian(self):
        for i in range(0, self.m):
            for j in range(0, self.m):
                if i!=j:
                    self.hamiltonian += self.hybrid * self.operators[i,j]
                else:
                    self.hamiltonian += (i) * (self.onsite) * self.operators[i,j]
        
        tmp = np.matrix( np.zeros( (self.dim , self.dim) ) )
        for i in range(0, self.m):
            for j in range(0, self.m):
                for k in range(0, self.m):
                    for l in range(0, self.m):
                        tmp = getQuartic(self,i,j,k,l)
                        if  i==j and k==l and k==j:
                            self.hamiltonian += (self.interequal) * tmp
                        else:
                            self.hamiltonian += (self.interdiff) * tmp
                        del tmp
    
    def initSpecLoHamiltonian(self):
        tmpspecops = quadraticArraySpecLo(self)
        for i in range(0, self.m):
            for j in range(0, self.m):
                if i!=j:
                    self.specLoHamiltonian += self.hybrid * tmpspecops[i,j]
                else:
                    self.specLoHamiltonian += (i) * (self.onsite) * tmpspecops[i,j]
        
        tmp = np.matrix( np.zeros( (self.specLoDim , self.specLoDim) ) )
        for i in range(0, self.m):
            for j in range(0, self.m):
                for k in range(0, self.m):
                    for l in range(0, self.m):
                        tmp = getQuarticSpec(tmpspecops,i,j,k,l)
                        if  i==j and k==l and k==j:
                            self.specLoHamiltonian += (self.interequal) * tmp
                        else:
                            self.specLoHamiltonian += (self.interdiff) * tmp
                        del tmp
        del tmpspecops
        
    def initSpecHiHamiltonian(self):
        tmpspecops = quadraticArraySpecHi(self)
        for i in range(0, self.m):
            for j in range(0, self.m):
                if i!=j:
                    self.specHiHamiltonian += self.hybrid * tmpspecops[i,j]
                else:
                    self.specHiHamiltonian += (i) * (self.onsite) * tmpspecops[i,j]
        
        tmp = np.matrix( np.zeros( (self.specHiDim , self.specHiDim) ) )
        for i in range(0, self.m):
            for j in range(0, self.m):
                for k in range(0, self.m):
                    for l in range(0, self.m):
                        tmp = getQuarticSpec(tmpspecops,i,j,k,l)
                        if  i==j and k==l and k==j:
                            self.specHiHamiltonian += (self.interequal) * tmp
                        else:
                            self.specHiHamiltonian += (self.interdiff) * tmp
                        del tmp
        del tmpspecops
    
    
    # The matrix already inherits the identity so step is just mutliplication
    # time evolution order given by order of the exponential series
    def initEvolutionMatrix(self, diagonalize=True):
        if self.order == 0:
            print('Warning - Time evolution of order 0 means no dynamics...')
        if (not np.allclose(self.hamiltonian.toarray(), np.conjugate(self.hamiltonian.toarray().T))):
            print('Warning - hamiltonian is not hermitian!')
        self.evolutionMatrix = spidentity(self.dim, dtype=self.datType, format='csr')
        
        for i in range(1, self.order + 1):
            self.evolutionMatrix += (-1j) ** i * (self.deltaT) ** i * self.hamiltonian ** i / factorial(i)
            
        self.evolutionMatrix = np.matrix(self.evolutionMatrix.toarray(), dtype=self.datType)
        if self.boolHamilStore:
            storeMatrix(self.hamiltonian.toarray(), './data/hamiltonian.txt', 1)
            storeMatrix(self.evolutionMatrix, './data/evolutionmatrix.txt', 1)
        self.evolutionMatrix = npmatrix_power(self.evolutionMatrix, self.evolStepDist)
        
        # Store hamiltonian eigenvalues
        if diagonalize:
            self.updateEigenenergies()
    # end
    
    # The matrix already inherits the identity so step is just mutliplication
    # time evolution order given by order of the exponential series
    # this one will be only in sparse container since it is meant for sparse matrix mult.
    #### IMPORTANT NOTE - complex conjugate will be needed for Green function ####
    #### FURTHER: need only 2*delta_T for green function, so added sq=True ####
    def initSpecLoEvolutionMatrix(self, diagonalize=False,conj=True,sq=True):
        if self.loOrder == 0:
            print('Warning - Time evolution of order 0 means no dynamics...')
        if (not np.allclose(self.specLoHamiltonian.toarray(), np.conjugate(self.specLoHamiltonian.toarray().T))):
            print('Warning - hamiltonian is not hermitian!')
        self.specLoEvolutionMatrix = spidentity(self.specLoDim, dtype=self.datType, format='csr')
        
        if conj:
            pre = (1j)
        else:
            pre = (-1j)
            
        for i in range(1, self.loOrder + 1):
            self.specLoEvolutionMatrix += pre ** i * (self.deltaT) ** i * self.specLoHamiltonian ** i / factorial(i)
        if diagonalize:
            self.updateLoEigenenergies()
        if sq:
            self.specLoEvolutionMatrix = self.specLoEvolutionMatrix ** 2
    # end
    
    # The matrix already inherits the identity so step is just mutliplication
    # time evolution order given by order of the exponential series
    # this one will be only in sparse container since it is meant for sparse matrix mult.
    def initSpecHiEvolutionMatrix(self, diagonalize=False,sq=True):
        if self.hiOrder == 0:
            print('Warning - Time evolution of order 0 means no dynamics...')
        if (not np.allclose(self.specHiHamiltonian.toarray(), np.conjugate(self.specHiHamiltonian.toarray().T))):
            print('Warning - hamiltonian is not hermitian!')
        self.specHiEvolutionMatrix = spidentity(self.specHiDim, dtype=self.datType, format='csr')
        
        for i in range(1, self.hiOrder + 1):
            self.specHiEvolutionMatrix += (-1j) ** i * (self.deltaT) ** i * self.specHiHamiltonian ** i / factorial(i)
        if diagonalize:
            self.updateHiEigenenergies()
        if sq:
            self.specHiEvolutionMatrix = self.specHiEvolutionMatrix ** 2
    # end
    
    def timeStep(self):
        self.state = npdot(self.evolutionMatrix, self.state)
    # end of timeStep
    
    def greenStoreState(self):
        self.stateSaves.append(self.state)
        self.timeSaves.append(self.evolTime)
        
    # approximate distributions in energy space - all parameters have to be set!
    # if skip is set to negative, the absolute value gives probability for finding a True in binomial
    def stateEnergy(self, muperc=[50], sigma=[1], phase=['none'], skip=[0], dist=['std'], peakamps=[1], skew=[0]):
        if self.eigInd == False:
            self.updateEigenenergies()
        
        self.state.fill(0)
        
        for i in range(0,len(muperc)):
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
                phaseArray = np.random.binomial(1,0.5,self.dim) * np.pi
            else:
                phaseArray = np.zeros(self.dim)
            
            if skip[i] < 0:
                skipArray = np.random.binomial(1,-1*skip[i],self.dim)
            elif skip[i] == 0:
                skipArray = np.zeros(self.dim)        
                skipArray[::1] = 1
            else:
                skipArray = np.zeros(self.dim)        
                skipArray[::int(skip[i])] = 1
            
            # mu is given in percent so get mu in energy space - also offsets are taken into account
            mu = self.eigVals[0] + (muperc[i] / 100) * (self.eigVals[-1] - self.eigVals[0])
            for k in range(0, self.dim):
                if skipArray[k]:
                    if dind == 1:
                        self.state[:, 0] += peakamps[i] * np.exp(1j * phaseArray[k]) * gaussian(self.eigVals[k], mu, sigma[i], norm=True, skw=skew[i]) * self.eigVects[:, k]
                    elif dind == 2:
                        self.state[:, 0] += peakamps[i] * np.exp(1j * phaseArray[k]) * rect(self.eigVals[k], mu, sigma[i], norm=False) * self.eigVects[:, k]
                    elif dind == 3:
                        self.state[k, 0] += peakamps[i] * np.exp(1j * phaseArray[k]) * tmpdist[k]
        del phaseArray
        del skipArray
        self.normalize(True)
    
    def normalize(self, initial=False):
        # note that the shape of the state vector is (dim,1) for reasons of matrix multiplication in numpy
        self.stateNorm = np.real(sqrt(npeinsum('ij,ij->j', self.state, np.conjugate(self.state))))[0]
        self.stateNormAbs *= self.stateNorm
        self.state /= self.stateNorm
        # do not store the new state norm - it is defined to be 1 so just store last norm value!
        # self.stateNorm = np.real(sqrt(npeinsum('ij,ij->j',self.state,np.conjugate(self.state))))[0]
        if bool(initial) == True:
            self.stateNormAbs = 1
            self.updateEigendecomposition()
            #store starting states used for green function
        if np.abs(self.stateNormAbs) > self.stateNormCheck:
            if self.stateNormCheck == 1e1:
                print('\n' + '### WARNING! ### state norm has been normalized by more than the factor 10 now!' + '\n' + 'Check corresponding plot if behavior is expected - indicator for numerical instability!' + '\n')
                self.stateNormCheck = 1e2
            else:
                self.closeFiles()
                self.plot()
                exit('\n' + 'Exiting - state norm has been normalized by more than the factor 100, numerical error is very likely.')
    # end of normalize    
    
    # note that - in principle - the expectation value can be complex! (though it shouldn't be)
    def expectValue(self, operator):
        if np.shape(operator) != (self.dim, self.dim):
            exit('Dimension of operator is', np.shape(operator), 'but', (self.dim, self.dim), 'is needed!')
        # will compute only the diagonal elements!
        return np.einsum('ij,ji->', self.densityMatrix, operator)
    
    def expectValueRed(self, operator):
        if np.shape(operator) != (self.dimRed, self.dimRed):
            exit('Dimension of operator is' +str(np.shape(operator)) + 'but' + str( (self.dimRed, self.dimRed) ) + 'is needed!')
        return np.trace(npdot(self.densityMatrixRed, operator))
    
    def updateEigenenergies(self):
        if not self.eigInd:
            self.eigVals, self.eigVects = la.eigh(self.hamiltonian.toarray())
            self.eigInd = True
    
    def updateLoEigenenergies(self):
        if not self.specLoEigInd:
            self.specLoEigVals, self.specLoEigVects = la.eigh(self.specLoHamiltonian.toarray())
            self.specLoEigInd = True

    def updateHiEigenenergies(self):
        if not self.specHiEigInd:
            self.specHiEigVals, self.specHiEigVects = la.eigh(self.specHiHamiltonian.toarray())
            self.specHiEigInd = True            
    
    #clear will delete the hamiltonian and (even more important) the corresponding eigenvectors
    def updateSpectralLo(self,clear=False):
        if not self.specLoInd:
            self.updateEigenenergies()
            self.updateLoEigenenergies()
            for i in range(0,self.m):
                self.specLoMatrix.append(np.dot(self.specLoEigVects.T , np.dot(getLoweringSpec(self,i).toarray(), self.eigVects)))
            if clear:
                del self.specLoHamiltonian
                del self.specHiEigVects
            self.specLoInd = True
    
    #clear will delete the hamiltonian and (even more important) the corresponding eigenvectors    
    def updateSpectralHi(self,clear=False):
        if not self.specHiInd:
            self.updateEigenenergies()
            self.updateHiEigenenergies()
            for i in range(0,self.m):
                self.specHiMatrix.append(np.dot(self.specHiEigVects.T , np.dot(getRaisingSpec(self,i).toarray(), self.eigVects)))
            if clear:
                del self.specHiHamiltonian
                del self.specHiEigVects
            self.specHiInd = True
    
    def storeSpectral(self,clear=False):
        for a in range(0,self.m):
            flo = open('./data/spectral/lo%i.txt' % (a), 'w')
            fhi = open('./data/spectral/hi%i.txt' % (a), 'w')
            for i in range(0,self.dim):
                for j in range(0,self.dim):
                    for k in range(0,self.specLoDim):
                        en = self.specLoEigVals[k] - (self.eigVals[i] + self.eigVals[j])/2
                        matel = self.specLoMatrix[a][k,i] * self.specLoMatrix[a][k,j]
                        flo.write('%i %i %i %.16e %.16e \n' % (i,j,k,en,matel))
                        en = (self.eigVals[i] + self.eigVals[j])/2 - self.specHiEigVals[k]
                        matel = self.specHiMatrix[a][k,i] * self.specHiMatrix[a][k,j]
                        fhi.write('%i %i %i %.16e %.16e \n' % (i,j,k,en,matel))
            fhi.close()
            flo.close()
        if clear:
            del self.specLoEigVals
            del self.specHiEigVals
            del self.specLoMatrix
            del self.specHiMatrix
            
    def storeSpectralDensityMatrix(self):
        self.updateEigenenergies()
        storeMatrix(np.dot(self.eigVects.T,np.dot(self.densityMatrix,self.eigVects)), './data/spectral/dm.txt', absOnly=False)
        
    ## will free the memory!!!
    def updateEigendecomposition(self,clear=True):
        if self.boolEngyStore:
            self.updateEigenenergies()
            # decomposition in energy space       
            tfil = open('./data/hamiltonian_eigvals.txt', 'w')  
            if self.boolDecompStore:
                tmpAbsSq = np.zeros(self.dim)
                for i in range(0, self.dim):
                    tmp = np.dot(self.eigVects[:, i], self.state[:, 0])                  
                    tmpAbsSq[i] = np.abs(tmp) ** 2
                    if tmpAbsSq[i] != 0:
                        tmpPhase = np.angle(tmp) / (2 * np.pi)  # angle in complex plane in units of two pi
                    else:
                        tmpPhase = 0
                    # occupation numbers of the eigenvalues
                    tfil.write('%i %.16e %.16e %.16e ' % (i, self.eigVals[i], tmpAbsSq[i] , tmpPhase))
                    for j in range(0, self.m):
                        tfil.write('%.16e ' % np.real(np.einsum('i,ii->', np.abs(self.eigVects[:, i]) ** 2, self.operators[j, j].toarray())))
                    tfil.write('\n')
            else:
                for i in range(0, self.dim):              
                    tfil.write('%i %.16e\n' % (i, self.eigVals[i]))
            tfil.close()   
            
            # decomposition in fock space
            sfil = open('./data/state.txt', 'w')
            for i in range(0, self.dim):               
                tmpAbsSqFck = np.abs(self.state[i, 0]) ** 2
                if tmpAbsSqFck != 0:
                    tmpPhase = np.angle(self.state[i, 0]) / (2 * np.pi)  # angle in complex plane in units of two pi
                else:
                    tmpPhase = 0
                # occupation numbers of the eigenvalues
                sfil.write('%i %.16e %.16e ' % (i, tmpAbsSqFck , tmpPhase))
                for j in range(0, self.m):
                    sfil.write('%i ' % self.basis[i, j])
                sfil.write('\n')
            sfil.close()
        
        if self.boolDiagExpStore or self.boolOccEnStore:
            self.updateEigenenergies()
            eivectinv = la.inv(np.matrix(self.eigVects.T))
        
        # expectation values in diagonal representation (ETH)
        if self.boolDiagExpStore:
            if not self.boolDecompStore:
                tmpAbsSq = np.zeros(self.dim)
                for i in range(0, self.dim):
                    tmp = np.dot(self.eigVects[:, i], self.state[:, 0])                  
                    tmpAbsSq[i] = np.abs(tmp) ** 2
            ethfil = open('./data/diagexpect.txt', 'w')
            for i in range(0, self.m):
                tmpocc = np.einsum('l,lj,jk,kl', tmpAbsSq, self.eigVects.T, self.operators[i, i].toarray(), eivectinv)
                ethfil.write('%i %.16e \n' % (i, tmpocc))
            ethfil.close()
        
        if self.boolOccEnStore:
            for i in range(0, self.m):
                storeMatrix(np.einsum('lj,jk,km -> lm', self.eigVects.T, self.operators[i, i].toarray(), eivectinv), './data/occ' + str(i) + '.txt', absOnly=0, stre=True, stim=False, stabs=False)
        
        if clear:
            # free the memory
            del self.eigVals
            del self.eigVects
            self.eigInd
            self.eigVals = []
            self.eigVects = []
            self.eigInd = False
        
    def updateEntropy(self):
        self.entropy = 0
        for el in la.eigvalsh(self.densityMatrix, check_finite=False):
            if np.imag(el) != 0:
                print('There is an Eigenvalue of the density matrix with imaginary part', np.imag(el))
            if np.real(el) > 0:
                self.entropy -= np.real(el) * nplog(np.real(el))
            if np.real(el) < -1e-7:
                print('Oh god, there is a negative eigenvalue smaller than 1e-7 ! Namely:', el)
    # end of updateEntropy
    
    def updateEntropyRed(self):
        if self.densityMatrixRed is None:
            return
        self.entropyRed = 0
        for el in la.eigvalsh(self.densityMatrixRed, check_finite=False):
            if np.imag(el) != 0:
                print('There is an Eigenvalue of the density matrix with imaginary part', np.imag(el))
            if np.real(el) > 0:
                self.entropyRed -= np.real(el) * nplog(np.real(el))
            if np.real(el) < 0 and np.abs(el) > 1e-7:
                print('Oh god, there is a negative eigenvalue smaller than 1e-7 ! Namely:', el)
    # end of updateEntropyRed
    
    def updateOccNumbers(self):
        for m in range(0, self.m):
            self.occNo[m] = np.real(self.expectValue(self.operators[m, m].toarray()))
    # end of updateOccNumbers
    
    def updateEnergy(self):
        self.energy = np.real(self.expectValue(self.hamiltonian.toarray()))
    # end of updateEnergy
    
    def evaluateGreen(self):
        # use dots for multithread!
        self.filGreen = open('./data/green.txt','w') #t, re, im
        
        tmpHiEvol = spidentity(self.specHiDim, dtype=self.datType, format='csr')
        tmpLoEvol = spidentity(self.specLoDim, dtype=self.datType, format='csr')
        tmpGreen = 0j
        
        saves = len(self.timeSaves)
        dt = self.timeSaves[1]

        #handle the i=0 case => equal time greens function is always -i:
        self.filGreen.write('%.16e ' % (0))    
        for ind in range(0,self.m):
            self.filGreen.write('%.16e %.16e ' % (0, -1))    
        self.filGreen.write(' \n')    
    
        for i in range(1,int(len(self.timeSaves)/2)):
            tmpHiEvol = tmpHiEvol * self.specHiEvolutionMatrix ## they need to be the squared ones!
            tmpLoEvol = tmpLoEvol * self.specLoEvolutionMatrix ## they need to be the squared ones!
            self.filGreen.write('%.16e ' % (dt*i)) 
            for m in range(0,self.m):
                tmpGreen = (self.stateSaves[ind+i].T.conjugate() * self.specRaising[m].T * tmpHiEvol * self.specRaising[m] * self.stateSaves[ind-i])[0] 
                tmpGreen -= (self.stateSaves[ind-i].T.conjugate() * self.specLowering[m].T * tmpLoEvol * self.specLowering[m] * self.stateSaves[ind+i])[0]
                ''' einsum version
                tmpGreen = np.einsum('ji,kl,lj -> j',self.stateSaves[ind+i].T.conjugate(), (self.specRaising[m].T * tmpHiEvol * self.specRaising[m]).toarray(), self.stateSaves[ind-i])[0] 
                tmpGreen -= np.einsum('ji,kl,lj -> j',self.stateSaves[ind-i].T.conjugate(),(self.specLowering[m].T * tmpLoEvol * self.specLowering[m]).toarray(), self.stateSaves[ind+i])[0]
                '''
                #note that the greensfunction is multiplied by -i, which is included in the writing below!
                #first number is real part, second imaginary
                self.filGreen.write('%.16e %.16e ' % (tmpGreen.imag, -1*tmpGreen.real))    
            self.filGreen.write(' \n')
            
        self.filGreen.close()
        '''
        for i in range(0,self.m):
            self.green[i] = -1j * (np.einsum('ij,ij -> j', self.state.T.conjugate(), (self.specRaising[i] * self.initStateHiKet[i]))[0] - np.einsum('ij,ij -> j',self.initStateLoBra[i], (self.specLowering[i] * self.state))[0])
        '''
    # update everything EXCEPT for total entropy and energy - they are only updated 100 times
    def updateEverything(self):
        self.evolTime += (self.evolStep - self.evolStepTmp) * self.deltaT
        self.evolStepTmp = self.evolStep
        self.normalize()
        if self.boolOnlyRed:
            self.reduceDensityMatrixFromState()
        else:
            self.updateDensityMatrix()
            self.reduceDensityMatrix()
            self.updateOccNumbers()
                    
        self.updateEntropyRed()
        
    ###### the magic of time evolution
    def evolve(self):
        # check if state has been normalized yet (or initialized)
        if self.stateNormAbs == 0:
            self.normalize(True)
        
        if self.boolDataStore:
            self.openFiles()
        
        self.evolStepTmp = self.evolStep
        stepNo = int(self.dataPoints / 100)
        dmFileFactor = self.dmFilesSkipFactor
        t0 = t1 = tm.time()  # time before iteration
        self.tavg = 0  # needed for estimation of remaining time
        print('Time evolution\n' + ' 0% ', end='')
        self.filProg = open('./data/progress.log', 'a')
        self.filProg.write('Time evolution\n' + ' 0% ')
        self.filProg.close()
        # percent loop
        for i in range(1, 11):
            # decimal loop
            for ii in range(1, 11):
                # need only dataPoints steps of size evolStepDist
                for j in range(0 , stepNo):
                    if self.boolDataStore:
                        self.updateEverything()
                        self.writeData()
                    if self.boolRetgreen:
                        self.greenStoreState()
                        
                    # ## Time Step!
                    self.timeStep()
                    self.evolStep += self.evolStepDist
                
                ######### TMP TMP TMP #########
                # store states for the greens function - temporarily only 100 times
                #if self.boolRetgreen:
                #    self.greenStoreState()
                
                # calculate total entropy and energy only 100 times, it is time consuming and only a check
                if self.boolTotalEnt:
                    self.updateEntropy()
                    self.filTotEnt.write('%.16e %.16e \n' % (self.evolTime, self.entropy))
                    
                if self.boolTotalEnergy:
                    self.updateEnergy()
                    self.filEnergy.write('%.16e %.16e \n' % (self.evolTime, self.energy))
                
                print('.', end='', flush=True)
                if self.dim > 1e3 or self.steps > 1e7 :
                    self.filProg = open('./data/progress.log', 'a')
                    self.filProg.write('.')
                    self.filProg.close()

            self.tavg *= int(self.evolStep - self.steps / 10)  # calculate from time/step back to unit: time
            self.tavg += tm.time() - t1  # add passed time
            self.tavg /= self.evolStep  # average over total number of steps
            t1 = tm.time()
            print(' 1-norm: ' + str(np.round(1 - self.stateNormAbs, 2)) + ' elapsed: ' + time_elapsed(t0, 60, 0) + " ###### eta: " + str(int(self.tavg * (self.steps - self.evolStep) / 60)) + "m " + str(int(self.tavg * (self.steps - self.evolStep) % 60)) + "s", "\n" + str(i * 10) + "% ", end='')
            self.filProg = open('./data/progress.log', 'a')
            self.filProg.write(' 1-norm: ' + str(1 - np.round(self.stateNormAbs, 2)) + ' elapsed ' + time_elapsed(t0, 60, 0) + " ###### eta: " + str(int(self.tavg * (self.steps - self.evolStep) / 60)) + "m " + str(int(self.tavg * (self.steps - self.evolStep) % 60)) + "s" + "\n" + str(i * 10) + "% ")
            self.filProg.close()
        
        if self.boolDataStore:
            self.updateEverything()   
            self.writeData()
        if self.boolRetgreen:
            self.greenStoreState()
        
        print('\n' + 'Time evolution finished after', time_elapsed(t0, 60), 'with average time/step of', "%.4e" % self.tavg)
        
        if self.boolDataStore:
            self.closeFiles()
    # end
    
    def writeData(self):
        if self.boolDMStore or self.boolDMRedStore:
            if dmFileFactor == self.dmFilesSkipFactor:
                dmFileFactor = 1
                if not self.boolOnlyRed:
                    if self.boolDMStore:
                        storeMatrix(self.densityMatrix, './data/density/densmat' + str(int(self.dmcount)) + '.txt')
                    if self.boolDMRedStore:
                        storeMatrix(self.densityMatrixRed, './data/red_density/densmat' + str(int(self.dmcount)) + '.txt')
                    self.dmcount += 1
            else:
                dmFileFactor += 1

        self.filEnt.write('%.16e %.16e \n' % (self.evolTime, self.entropyRed))
        self.filNorm.write('%.16e %.16e %.16e \n' % (self.evolTime, self.stateNorm, self.stateNormAbs))
        self.filOcc.write('%.16e ' % self.evolTime)
        for m in range(0, self.m):
            self.filOcc.write('%.16e ' % self.occNo[m])
        self.filOcc.write('\n')
            
    def openFiles(self):
        self.filEnt = open('./data/entropy.txt', 'w')
        self.filTotEnt = open('./data/total_entropy.txt', 'w')
        self.filNorm = open('./data/norm.txt', 'w')
        self.filOcc = open('./data/occupation.txt', 'w')
        self.filEnergy = open('./data/energy.txt', 'w')
        self.filProg = open('./data/progress.log', 'w')
        self.filProg.close()

    #close all files
    def closeFiles(self):
        self.filEnt.close()
        self.filTotEnt.close()
        self.filNorm.close()
        self.filOcc.close()
        self.filEnergy.close()
            
    def plotDMAnimation(self, stepSize):
        ep.plotDensityMatrixAnimation(self.steps, self.deltaT , self.dmFiles , stepSize, framerate=self.dmFilesFPS)
    # end of plotDMAnimation
    
    def plotDMRedAnimation(self, stepSize):
        ep.plotDensityMatrixAnimation(self.steps, self.deltaT , self.dmFiles , stepSize, 1, framerate=self.dmFilesFPS)
    # end of plotDMAnimation
    
    def plotData(self):
        ep.plotData(self)
    # end of plotData
    
    def plotHamiltonian(self):
        ep.plotHamiltonian()
    # end of plotHamiltonian
    
    def plotOccEnbasis(self):
        ep.plotOccs(self)
    
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
        if self.boolClear:
            prepFolders(True)

    def clearDensityData(self):
        prepFolders(True)
    
def prepFolders(clearbool=0):
    # create the needed folders
    if not os.path.exists("./data/"):
        os.mkdir("./data/")
        print("Creating ./data Folder since it didn't exist")
    if not os.path.exists("./data/density/"):
        os.mkdir("./data/density/")
        print("Creating ./data/density Folder since it didn't exist")
    if not os.path.exists("./data/red_density/"):
        os.mkdir("./data/red_density/")
        print("Creating ./data/red_density Folder since it didn't exist")
    if not os.path.exists("./data/spectral/"):
        os.mkdir("./data/spectral/")
        print("Creating ./data/spectral Folder since it didn't exist")
    if not os.path.exists("./plots/"):
        os.mkdir("./plots/")
        print("Creating ./plts Folder since it didn't exist")
    # remove the old stuff
    if clearbool:
        if os.path.isfile("./data/density/densmat0.txt") == True:
            for root, dirs, files in os.walk('./data/density/', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            print("Cleared density folder")
        
        if os.path.isfile("./data/red_density/densmat0.txt") == True:
            for root, dirs, files in os.walk('./data/red_density/', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            print("Cleared reduced density folder")
          
# calculate the number of Coefficients
def dimOfBasis(N, m):
    return np.uint32(binom(N + m - 1, N))

def fillBasis(basis, N, m, offset=0):
    if m != 1:
        counter = [offset]
        for n in range(0, m - 1):
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
        tmp *= (1 + erf(skw * (x - mu) / (sigm * sqrt(2))))
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

def basisOffsets(N, m):
    offsets = np.zeros((N + 2) , dtype=np.int32)
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
    for i in range(0, N + 1):
        fillBasis(basis, i, m, offsets[i])
    # end
        
# filling arrays for l != m-1
def a(N, m, l, basis, Nsys, msys, counter):
    if m == msys - l:
        for n in range(0, N + 1):
            nn = 0
            while nn < dimOfBasis(n, m - 1):
                basis[counter[0]][l] = int(N - n)
                counter[0] += 1
                nn += 1
    else:
        for n in reversed(range(0, N + 1)):
            a(N - n, m - 1, l, basis, Nsys, msys, counter)
    # end

# filling arrays for l == m-1 (order is other way round)
def am(N, m, l, basis, Nsys, msys, counter):
    if m == msys:
        am(N, m - 1, l, basis, Nsys, msys, counter)
    elif m == msys - l:
        for n in reversed(range(0, N + 1)):
            basis[counter[0]][l] = int(N - n)
            counter[0] += 1
    else:
        for n in reversed(range(0, N + 1)):
            am(N - n, m - 1, l, basis, Nsys, msys, counter)
    # end

def basis2dict(basis, dim):
    # create an empty dictionary with states in occupation number repres. as tuples being the keys
    tup = tuple(tuple(el) for el in basis)
    dRet = dict.fromkeys(tup)
    # for correct correspondence go through the basis tuple and put in the vector-number corresponding to the given tuple
    for i in range(0, dim):
        dRet[tup[i]] = i
    return dRet

# note that all the elements here are sparse matrices! one has to use .toarray() to get them done correctly
def quadraticArray(sysVar):
    retArr = np.empty((sysVar.m , sysVar.m), dtype=csr_matrix)
    # off diagonal
    for i in range(0, sysVar.m):
        for j in range(0, i):
            retArr[i, j] = getQuadratic(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(0, sysVar.m):
        retArr[i, i] = getQuadratic(sysVar, i, i)
    return retArr

# note that all the elements here are sparse matrices! one has to use .toarray() to get them done correctly
# please also note the index shift - from low to high but without the traced out, e.g.
# m=4 trace out 1,2 -> 0,1 of reduced array corresponds to level 0 and 4 of whole system
def quadraticArrayRed(sysVar):
    retArr = np.empty((sysVar.mRed , sysVar.mRed), dtype=csr_matrix)
    # off diagonal
    for i in range(0, sysVar.mRed):
        for j in range(0, i):
            retArr[i, j] = getQuadraticRed(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(0, sysVar.mRed):
        retArr[i, i] = getQuadraticRed(sysVar, i, i)
    return retArr

def quadraticArraySpecLo(sysVar):
    retArr = np.empty((sysVar.m , sysVar.m), dtype=csr_matrix)
    # off diagonal
    for i in range(0, sysVar.m):
        for j in range(0, i):
            retArr[i, j] = getQuadraticSpecLo(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(0, sysVar.m):
        retArr[i, i] = getQuadraticSpecLo(sysVar, i, i)
    return retArr

def quadraticArraySpecHi(sysVar):
    retArr = np.empty((sysVar.m , sysVar.m), dtype=csr_matrix)
    # off diagonal
    for i in range(0, sysVar.m):
        for j in range(0, i):
            retArr[i, j] = getQuadraticSpecHi(sysVar, i, j)
            retArr[j, i] = retArr[i, j].transpose()
    # diagonal terms
    for i in range(0, sysVar.m):
        retArr[i, i] = getQuadraticSpecHi(sysVar, i, i)
    return retArr

# quadratic term in 2nd quantization for transition from m to l -> fills zero initialized matrix
# matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadratic(sysVar, l, m):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.m), dtype=np.int)
    for el in sysVar.basis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row = np.append(row, sysVar.basisDict[tuple(tmp)])
            col = np.append(col, sysVar.basisDict[tuple(el)])
            data = np.append(data, np.float64(sqrt(el[m]) * sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.dim , sysVar.dim) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat

# quadratic term in 2nd quantization for transition from m to l -> fills zero initialized matrix
# matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadraticRed(sysVar, l, m):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.mRed), dtype=np.int)
    for el in sysVar.basisRed:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row = np.append(row, sysVar.basisDictRed[tuple(tmp)])
            col = np.append(col, sysVar.basisDictRed[tuple(el)])
            data = np.append(data, np.float64(sqrt(el[m]) * sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.dimRed , sysVar.dimRed) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat

def getQuadraticSpecLo(sysVar, l, m):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.m), dtype=np.int)
    for el in sysVar.specLoBasis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row = np.append(row, sysVar.specLoBasisDict[tuple(tmp)])
            col = np.append(col, sysVar.specLoBasisDict[tuple(el)])
            data = np.append(data, np.float64(sqrt(el[m]) * sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.specLoDim , sysVar.specLoDim) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat

def getQuadraticSpecHi(sysVar, l, m):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.m), dtype=np.int)
    for el in sysVar.specHiBasis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row = np.append(row, sysVar.specHiBasisDict[tuple(tmp)])
            col = np.append(col, sysVar.specHiBasisDict[tuple(el)])
            data = np.append(data, np.float64(sqrt(el[m]) * sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.specHiDim , sysVar.specHiDim) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat


# array elements are NO matrix! just numpy array!
# This will take very long to create and use up much memory, please consider doing it on the fly only for needed elements.           
def quarticArray(sysVar):
    retArr = np.empty((sysVar.m , sysVar.m, sysVar.m, sysVar.m), dtype=csr_matrix)
    # TODO: use transpose property
    for k in range(0, sysVar.m):
        for l in range(0, sysVar.m):
            for m in range(0, sysVar.m):
                for n in range(0, sysVar.m):
                        retArr[k, l, m, n] = getQuartic(sysVar, k, l, m, n)
    return retArr

# array elements are NO matrix! just numpy array!
# This will take very long to create and use up much memory, please consider doing it on the fly only for needed elements.           
def quarticArrayRed(sysVar):
    retArr = np.empty((sysVar.mRed , sysVar.mRed, sysVar.mRed, sysVar.mRed), dtype=csr_matrix)
    # TODO: use transpose property
    for k in range(0, sysVar.mRed):
        for l in range(0, sysVar.mRed):
            for m in range(0, sysVar.mRed):
                for n in range(0, sysVar.mRed):
                        retArr[k, l, m, n] = getQuarticRed(sysVar, k, l, m, n)
    return retArr

def getQuartic(sysVar, k, l, m, n):
    if l != m:
        return (sysVar.operators[k, m] * sysVar.operators[l, n]).copy()
    else:
        return ((sysVar.operators[k, m] * sysVar.operators[l, n]) - sysVar.operators[k, n]).copy()

def getQuarticRed(sysVar, k, l, m, n):
    if l != m:
        return (getQuadraticRed(sysVar, k, m) * getQuadraticRed(sysVar, l, n)).copy()
    else:
        return ((getQuadraticRed(sysVar, k, m) * getQuadraticRed(sysVar, l, n)) - getQuadraticRed(sysVar, k, n)).copy()

def getQuarticSpec(quadops, k, l, m, n):
    if l != m:
        return (quadops[k, m] * quadops[l, n]).copy()
    else:
        return ((quadops[k, m] * quadops[l, n]) - quadops[k, n]).copy()

# destruction operator
def getLoweringSpec(sysVar,l):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.m), dtype=np.int)
    for el in sysVar.basis:
        if el[l] != 0:
            tmp = el.copy()
            tmp[l] -= 1
            row = np.append(row, sysVar.specLoBasisDict[tuple(tmp)])
            col = np.append(col, sysVar.basisDict[tuple(el)])
            data = np.append(data, np.float64(sqrt(el[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.specLoDim , sysVar.dim) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat

# creation operator
def getRaisingSpec(sysVar,l):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.m), dtype=np.int)
    for el in sysVar.basis:
        tmp = el.copy()
        tmp[l] += 1
        row = np.append(row, sysVar.specHiBasisDict[tuple(tmp)])
        col = np.append(col, sysVar.basisDict[tuple(el)])
        data = np.append(data, np.float64(sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.specHiDim , sysVar.dim) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat

# inverse of creation operator (have to multiply from left...)
def getRaisingSpecInv(sysVar,l):
    data = np.zeros(0, dtype=np.float64)
    row = np.zeros(0, dtype=np.float64)
    col = np.zeros(0, dtype=np.float64)
    tmp = np.zeros((sysVar.m), dtype=np.int)
    for el in sysVar.basis:
        tmp = el.copy()
        tmp[l] += 1
        col = np.append(col, sysVar.specHiBasisDict[tuple(tmp)])
        row = np.append(row, sysVar.basisDict[tuple(el)])
        data = np.append(data, np.float64(1/sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row, col)), shape=(sysVar.dim, sysVar.specHiDim) , dtype=np.float64).tocsr()
    del row, col, data, tmp
    return retmat

def time_elapsed(t0, divider, decimals=0):
    t_el = tm.time() - t0
    if divider == 60:
        t_min = t_el // 60
        t_sec = t_el % 60
        return (str(int(t_min)) + "m " + str(int(t_sec)) + "s")
    else: 
        t_sec = t_el / divider
        return str(round(t_sec, decimals)) + "s"

# stores the matrix of dimension sysvar[2] in a file
def storeMatrix(mat, fil, absOnly=0, stre=True, stim=True, stabs=True):
    matDim = np.shape(mat)[0]
    if absOnly == 0:
        # assume dot + 3 letter ending e.g. .txt
        fname = fil[:-4]
        fend = fil[-4:]
        if stabs:
            f = open(fil, 'w')
        if stim:
            fimag = open(fname + '_im' + fend, 'w')
        if stre:
            freal = open(fname + '_re' + fend, 'w')
        for n in range(0, matDim):
            for nn in range(0, matDim - 1):
                if stabs:
                    f.write('%.16e ' % np.abs(mat[(n, nn)]))
                if stim:
                    fimag.write('%.16e ' % np.imag(mat[(n, nn)]))
                if stre:
                    freal.write('%.16e ' % np.real(mat[(n, nn)]))
            if stabs:
                f.write('%.16e\n' % np.abs(mat[(n, matDim - 1)]))
            if stim:
                fimag.write('%.16e\n' % np.imag(mat[(n, matDim - 1)]))
            if stre:
                freal.write('%.16e\n' % np.real(mat[(n, matDim - 1)]))
        if stabs:
            f.close()
        if stim:
            fimag.close()
        if stre:
            freal.close()
    else:
        f = open(fil, 'w')
        # assume dot + 3 letter ending e.g. .txt
        for n in range(0, matDim):
            for nn in range(0, matDim - 1):
                f.write('%.16e ' % np.abs(mat[(n, nn)]))
            f.write('%.16e\n' % np.abs(mat[(n, matDim - 1)]))
        f.close()  
        
