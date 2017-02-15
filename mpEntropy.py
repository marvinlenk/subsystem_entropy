import numpy as np
from numpy import dot as npdot
from scipy.special import binom
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse import identity as spidentity
import scipy.linalg as la
from numpy import einsum as npeinsum
import time as tm
import os, configparser
import entPlot as ep
from numpy import sqrt
from numpy import log as nplog
from numpy.linalg import matrix_power as npmatrix_power

#manyparticle system class
class mpSystem:
    #N = total particle number, m = number of states in total, redStates = array of state indices to be traced out
    def __init__(self,cFile="config.ini",dtType=np.complex128):
        self.confFile = cFile
        prepFolders(0,self.confFile)
        self.loadConfig()
        ###### system variables
        self.datType = dtType
        self.dim = dimOfBasis(self.N,self.m) 
        self.basis = np.zeros( (self.dim , self.m) , dtype = np.int)
        fillBasis(self.basis, self.N, self.m)
        self.basisDict = basis2dict(self.basis, self.dim)
        #note that there is an additional dimension there! needed for fast multiplication algorithm
        self.state = np.zeros( (self.dim,1) , dtype = self.datType)
        # parameter for storing in file
        self.stateNorm = 0
        self.stateNormAbs = 1
        self.stateNormCheck = 1e1 #check if norm has been supressed too much
        self.densityMatrix = np.zeros( (self.dim , self.dim) , dtype = self.datType) 
        self.entropy = 0
        self.energy = 0
        self.operators = quadraticArray(self)
        self.occNo = np.zeros(self.m,dtype=np.float64)
        self.transVal = np.zeros( len(self.transExp) , dtype = np.float64)
        # hamiltonian - initialized with zeros (note - datatype is not! complex)
        self.hamiltonian = coo_matrix(np.zeros((self.dim,self.dim)),shape=(self.dim,self.dim),dtype=np.float64).tocsr()
        # matrix for time evolution - initially empty
        self.evolutionMatrix = None
        # iteration step
        self.evolStep = 0
        self.evolStepTmp = 0
        self.evolTime = 0
        self.tavg = 0    #needed for estimation of remaining time
        ###### variables for the partial trace algorithm
        self.mRed = self.m - len(self.kRed)
        self.mRedComp = len(self.kRed)
        self.entropyRed = 0
        
        # mask selects only not traced out states
        self.mask = np.ones( (self.m) , dtype = bool)
        for k in self.kRed:
            self.mask[k] = False

        if self.mRedComp == 0:
            self.dimRed = 0
            self.offsetsRed = None
            self.basisRed = None
            self.dimRedComp = self.dim
            self.offsetsRedComp = np.zeros( (self.N+2), dtype=np.int32)
            self.offsetsRedComp[-1] = self.dim
            self.basisRedComp = self.basis
            self.densityMatrixRed = None
        else:
            #particle number bound from above -> dim is that of 1 state more but with particle number conservation
            self.dimRed = dimOfBasis( self.N , (self.mRed + 1) )
            self.offsetsRed = basisOffsets( self.N, self.mRed)
            self.basisRed = np.zeros( (self.dimRed , self.mRed) , dtype = np.int)
            fillReducedBasis(self.basisRed, self.N, self.mRed, self.offsetsRed)
            self.dimRedComp = dimOfBasis ( self.N , (self.mRedComp + 1) )
            self.offsetsRedComp = basisOffsets(self.N, self.mRedComp)
            self.basisRedComp = np.zeros( (self.dimRedComp , self.mRedComp) , dtype = np.int)
            fillReducedBasis(self.basisRedComp, self.N, self.mRedComp, self.offsetsRedComp)
            self.densityMatrixRed = np.zeros( (self.dimRed , self.dimRed) , dtype = self.datType)
            self.iteratorRed = np.zeros( (0 , 4), dtype = np.int32 )
            self.initIteratorRed() 
    #end of init
    
    ###### reading from config file
    def loadConfig(self):
        configParser = configparser.RawConfigParser()  
        configParser.read('./'+self.confFile)
        ### system parameters
        self.N = int(configParser.getfloat('system','N'))
        self.m = int(configParser.getfloat('system','m'))
        self.kRed = configParser.get('system','kred').split(',')
        if len(self.kRed[0]) == 0:
            self.kRed = []
        else:
            self.kRed = [int(el) for el in self.kRed]
        ### iteration parameters
        self.steps = int(configParser.getfloat('iteration','steps'))
        self.deltaT = np.float64(configParser.getfloat('iteration','deltaT'))
        ### file management
        self.dataPoints = int(configParser.getfloat('filemanagement','datapoints'))
        self.dmFilesSkipPoints = int(configParser.getfloat('filemanagement','dmfile_skippoints'))
        self.boolClear = configParser.getboolean('filemanagement','clear')
        self.boolDataStore = configParser.getboolean('filemanagement','datastore')
        self.boolDMStore = configParser.getboolean('filemanagement','dmstore') 
        self.transExp = [tuple( [int(el[0]), int(el[1])] ) for el in configParser.get('filemanagement','transexp').split(',')]
        ### calculation-parameters
        self.boolOnlyRed = configParser.getboolean('calcparams','onlyreduced')
        self.boolTotalEnt = configParser.getboolean('calcparams','totalentropy')
        ### plotting booleans and parameters
        self.boolPlotData = configParser.getboolean('plotbools','plotdata')
        self.boolPlotHamiltonian = configParser.getboolean('plotbools','plothamiltonian')
        self.boolPlotDMAnimation = configParser.getboolean('plotbools','plotdens')
        self.boolPlotDMRedAnimation = configParser.getboolean('plotbools','plotdensred')
        self.dmFilesStepSize = configParser.getint('plotbools','dmstepsize')
        
        self.evolStepDist = int(self.steps/self.dataPoints)
        if self.evolStepDist < 100:
            self.steps += (100 - self.evolStepDist) * self.steps
            self.evolStepDist = 100
            print('Number of steps must be at least by factor 100 larger than datapoints! Fixed this.')
        self.dmFiles = self.dataPoints / self.dmFilesSkipPoints    
        
        if self.dataPoints > self.steps:
            self.dataPoints = self.steps
            print('Number of data points was larger than number of steps - think again!')
        
    ###### Methods:
    def updateDensityMatrix(self):
        self.densityMatrix = np.outer( self.state[:,0], np.conjugate(self.state[:,0]) )
    #end of updateDensityMatrix
    
    def initIteratorRed(self):
        el1 = np.zeros( (self.m) , dtype=np.int)
        el2 = np.zeros( (self.m) , dtype=np.int)
        for i in reversed(range(0,self.N+1)):
            for j in range(self.offsetsRed[i],self.offsetsRed[i-1]):
                for jj in range(j,self.offsetsRed[i-1]):
                    for k in range(self.offsetsRedComp[self.N - i], self.offsetsRedComp[self.N - i - 1]):
                        el1[self.mask] = self.basisRed[j]
                        el1[~self.mask] = self.basisRedComp[k]
                        el2[self.mask] = self.basisRed[jj]
                        el2[~self.mask] = self.basisRedComp[k]
                        self.iteratorRed = np.append(self.iteratorRed,[[j,jj,self.basisDict[tuple(el1)],self.basisDict[tuple(el2)]]],axis=0)
    #end of initTest
    
    def reduceDensityMatrix(self):
        if self.densityMatrixRed is None:
            return
        self.densityMatrixRed.fill(0)
        for el in self.iteratorRed:
            self.densityMatrixRed[el[0],el[1]] += self.densityMatrix[el[2],el[3]]
            if el[0] != el[1]:
                self.densityMatrixRed[el[1],el[0]] += self.densityMatrix[el[3],el[2]]
    
    def reduceDensityMatrixFromState(self):
        stTmp = np.conjugate(self.state)
        if self.densityMatrixRed is None:
            return
        self.densityMatrixRed.fill(0)
        for el in self.iteratorRed:
            self.densityMatrixRed[el[0],el[1]] += self.densityMatrix[el[2],el[3]]
            if el[0] != el[1]:
                self.densityMatrixRed[el[1],el[0]] += self.state[el[3],0] * stTmp[el[2],0]
    #end of reduceDensityMatrixFromState
        
    #Runge Kutta of order 1 to 4, standard is 4th Order - the matrix already inherits the identity so step is just mutliplication
    def initEvolutionMatrix(self, order = 4):
        if (not np.allclose(self.hamiltonian.toarray(), np.conjugate(self.hamiltonian.toarray().T))):
            print('Warning - hamiltonian is not hermitian!')
        if order > 4:
            exit('Only Runge-Kutta to order 4 is supported')
        self.evolutionMatrix = spidentity(self.dim, dtype=self.datType, format='csr')
        if order >= 1:
            self.evolutionMatrix += (-1j * self.deltaT) * self.hamiltonian
        if order >= 2:
            self.evolutionMatrix += (-1 * self.deltaT**2 / 2.0) * self.hamiltonian **2
        if order >= 3:
            self.evolutionMatrix += (1j * self.deltaT**3 / 6.0) * self.hamiltonian **3
        if order == 4:
            self.evolutionMatrix += (1 * self.deltaT**4 / 24.0) * self.hamiltonian **4
        self.evolutionMatrix = np.matrix(self.evolutionMatrix.toarray(), dtype = self.datType)
        storeMatrix(self.hamiltonian.toarray(), './data/hamiltonian.txt', 1)
        storeMatrix(self.evolutionMatrix, './data/evolutionmatrix.txt', 1)
        self.evolutionMatrix = npmatrix_power(self.evolutionMatrix,self.evolStepDist)
    #end
    
    def timeStep(self):
        self.state = npdot(self.evolutionMatrix, self.state)
    #end of timeStep

    def normalize(self,initial=False):
        self.stateNorm = np.real(sqrt(npeinsum('ij,ij->j',self.state,np.conjugate(self.state))))[0]
        self.stateNormAbs *= self.stateNorm
        self.state /= self.stateNorm
        #do not store the new state norm - it is defined to be 1 so just store last norm value!
        #self.stateNorm = np.real(sqrt(npeinsum('ij,ij->j',self.state,np.conjugate(self.state))))[0]
        if bool(initial) == True:
            self.stateNormAbs = 1
        if np.abs(self.stateNormAbs) > self.stateNormCheck:
            if self.stateNormCheck == 1e1:
                print('\n'+'### WARNING! ### state norm has been normalized by more than the factor 10 now!'+'\n'+'Check corresponding plot if behavior is expected - indicator for numerical instability!'+'\n')
                self.stateNormCheck = 1e2
            else:
                self.filEnt.close()
                self.filNorm.close()
                self.filOcc.close()
                self.filEnergy.close()
                self.plot()
                exit('\n'+'Exiting - state norm has been normalized by more than the factor 100, numerical error is very likely.')
    #end of normalize    
    
    #note that - in principle - the expectation value can be complex! (though it shouldn't be)
    def expectValue(self, operator):
        if np.shape(operator) != ( self.dim, self.dim ):
            exit('Dimension of operator is',np.shape(operator),'but',( self.dim, self.dim ), 'is needed!')
        #will compute only the diagonal elements!
        return np.einsum('ij,ji->', self.densityMatrix, operator)
    
    def expectValueRed(self, operator):
        if np.shape(operator) != ( self.dimRed, self.dimRed ):
            exit('Dimension of operator is',np.shape(operator),'but',( self.dimRed, self.dimRed ), 'is needed!')
        return np.trace( npdot(self.densityMatrixRed, operator) )
    
    def updateEntropy(self):
        self.entropy = 0
        for el in la.eigvalsh(self.densityMatrix,check_finite=False):
            if np.imag(el) != 0:
                print('There is an Eigenvalue of the density matrix with imaginary part', np.imag(el))
            if np.real(el) > 0:
                self.entropy -= np.real(el) * nplog(np.real(el))
            if np.real(el) < -1e-7:
                print('Oh god, there is a negative eigenvalue smaller than 1e-7 ! Namely:', el)
    #end of updateEntropy
    
    def updateEntropyRed(self):
        if self.densityMatrixRed is None:
            return
        self.entropyRed = 0
        for el in la.eigvalsh(self.densityMatrixRed,check_finite=False):
            if np.imag(el) != 0:
                print('There is an Eigenvalue of the density matrix with imaginary part', np.imag(el))
            if np.real(el) > 0:
                self.entropyRed -= np.real(el) * nplog(np.real(el))
            if np.real(el) < 0 and np.abs(el) > 1e-7:
                print('Oh god, there is a negative eigenvalue smaller than 1e-7 ! Namely:', el)
    #end of updateEntropyRed
    
    def updateOccNumbers(self):
        for m in range( 0, self.m ):
            self.occNo[m] = np.real(self.expectValue(self.operators[m,m].toarray()))
    #end of updateOccNumbers
    
    def updateTransNumbers(self):
        for tup in self.transExp:
            return
            
    
    def updateEnergy(self):
        self.energy = np.real(self.expectValue(self.hamiltonian.toarray()))
    #end of updateEnergy
    
    def updateEverything(self):
        self.evolTime += ( self.evolStep - self.evolStepTmp ) * self.deltaT
        self.evolStepTmp = self.evolStep
        self.normalize()
        if self.boolOnlyRed:
            self.reduceDensityMatrixFromState()
        else:
            self.updateDensityMatrix()
            self.reduceDensityMatrix()
            if self.boolTotalEnt:
                self.updateEntropy()
            self.updateOccNumbers()
            self.updateEnergy()
        
        self.updateEntropyRed()
        
    ###### the magic of time evolution
    def evolve(self):
        if self.boolDataStore:
            self.filEnt = open('./data/entropy.txt','w')
            self.filNorm = open('./data/norm.txt','w')
            self.filOcc = open('./data/occupation.txt','w')
            self.filEnergy = open('./data/energy.txt','w')
            self.filProg = open('./data/progress.log','w')
            self.filProg.close()
            
        self.evolStepTmp = self.evolStep
        stepNo = int(self.dataPoints/100)
        dmFileSkip = self.dmFilesSkipPoints
        t0 = t1 = tm.time() #time before iteration
        self.tavg = 0    #needed for estimation of remaining time
        print('Time evolution\n'+'0%  ',end='')
        self.filProg = open('./data/progress.log','a')
        self.filProg.write('Time evolution\n'+'0%  ')
        self.filProg.close()
        #percent loop
        for i in range(1,11):
            #decimal loop
            for ii in range(1,11):
                #need only dataPoints steps of size evolStepDist
                for j in range(0 , stepNo):
                    if self.boolDataStore:
                        self.updateEverything()
                        
                        if self.boolDMStore:
                            if dmFileSkip == self.dmFilesSkipPoints:
                                dmFileSkip = 0
                                if not self.boolOnlyRed:
                                    storeMatrix(self.densityMatrix,'./data/density/densmat'+str(self.evolStep)+'.txt')
                            
                                    storeMatrix(self.densityMatrixRed,'./data/red_density/densmat'+str(self.evolStep)+'.txt')
                            else:
                                dmFileSkip += 1

                        self.filEnt.write( str(self.evolTime) + ' ' + str(self.entropy) + ' ' + str(self.entropyRed) + '\n' )
                        self.filNorm.write( str(self.evolTime) + ' ' + str(self.stateNorm) + ' ' + str(self.stateNormAbs) + '\n' )
                        self.filOcc.write(str(self.evolTime)+ ' ')
                        for m in range(0,self.m):
                            self.filOcc.write(str(self.occNo[m]) + ' ')
                        self.filOcc.write('\n')
                        self.filEnergy.write( str(self.evolTime) + ' ' + str(self.energy) + '\n' )
                            
                    ### Time Step!
                    self.timeStep()
                    self.evolStep += self.evolStepDist
                    #print(i,ii,j)
                    
                print('.',end='',flush=True)
                if self.dim > 1e3 or self.steps > 1e7 :
                    self.filProg = open('./data/progress.log','a')
                    self.filProg.write('.')
                    self.filProg.close()

            self.tavg *= int(self.evolStep - self.steps/10) #calculate from time/step back to unit: time
            self.tavg += tm.time() - t1  #add passed time
            self.tavg /= self.evolStep   #average over total number of steps
            t1 = tm.time()
            print(' norm total: '+str(np.round(self.stateNormAbs,2))+ ' ' + time_elapsed(t0,60,0) + " ######## etr: " + str(int(self.tavg*(self.steps-self.evolStep)/60)) + "m " + str(int(self.tavg*(self.steps-self.evolStep)%60)) + "s", "\n" + str(i*10) + "% ",end='')
            self.filProg = open('./data/progress.log','a')
            self.filProg.write(' norm total: '+str(np.round(self.stateNormAbs,2))+ ' ' + time_elapsed(t0,60,0) + " ######## etr: " + str(int(self.tavg*(self.steps-self.evolStep)/60)) + "m " + str(int(self.tavg*(self.steps-self.evolStep)%60)) + "s"+ "\n" + str(i*10) + "% ")
            self.filProg.close()
        print('\n'+'Time evolution finished after',time_elapsed(t0, 60), 'with average time/step of',"%.4e" % self.tavg)
        
        if self.boolDataStore:
            self.filEnt.close()
            self.filNorm.close()
            self.filOcc.close()
            self.filEnergy.close()
    #end
    
    def plotDMAnimation(self, stepSize):
        ep.plotDensityMatrixAnimation(self.steps, self.deltaT ,self.dmFiles , stepSize)
    #end of plotDMAnimation
    
    def plotDMRedAnimation(self, stepSize):
        ep.plotDensityMatrixAnimation(self.steps, self.deltaT ,self.dmFiles , stepSize, 1)
    #end of plotDMAnimation
    
    def plotData(self):
        ep.plotData(self)
    #end of plotData
    
    def plotHamiltonian(self):
        ep.plotHamiltonian()
    #end of plotMatrix
    
    def plot(self):
        if self.boolPlotData:
            self.plotData()
        if self.boolPlotHamiltonian:
            self.plotHamiltonian()
        if self.boolPlotDMAnimation:
            self.plotDMAnimation(self.dmFilesStepSize)
        if self.boolPlotDMRedAnimation:
            self.plotDMRedAnimation(self.dmFilesStepSize)
        if self.boolClear:
            prepFolders(True)
            
    def clearDensityData(self):
        prepFolders(True)
    
def prepFolders(clearbool = 0, cFile = "config.ini"):
    #create the needed folders
    if not os.path.exists("./data/"):
        os.mkdir("./data/")
        print("Creating ./data Folder since it didn't exist")
    if not os.path.exists("./data/density/"):
        os.mkdir("./data/density/")
        print("Creating ./data/density Folder since it didn't exist")
    if not os.path.exists("./data/red_density/"):
        os.mkdir("./data/red_density/")
        print("Creating ./data/red_density Folder since it didn't exist")
    if not os.path.exists("./plots/"):
        os.mkdir("./plots/")
        print("Creating ./plts Folder since it didn't exist")
    if os.path.isfile("./"+cFile) == False:
        confFile = open('./'+cFile,'w')
        confFile.write('[system]\n'+'N = 10\n'+'m = 4'+'\n'+'kred = '+'\n\n')
        confFile.write('[iteration]\n'+'deltaT = 1e-5\n'+'steps = 1e5\n\n')
        confFile.write('[filemanagement]\n'+'datapoints = 1e2\n'+'dmfiles = 1e2\n'+'clear = False\n'+'datastore = True\n'+'dmstore = False\n\n')
        confFile.write('[calcparams]\n'+'onlyreduced = False\n'+'totalentropy = False\n\n')
        confFile.write('[plotbools]\n'+'plotdata = True\n'+'plothamiltonian = True\n'+'plotdens = False\n'+'plotdensred = False\n'+'dmstepsize = 2')
        confFile.close()
        print('Created generic config file with name',cFile)
    #remove the old stuff
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
          
#calculate the number of Coefficients
def dimOfBasis(N,m):
    return np.uint32(binom(N+m-1,N))

def fillBasis(basis,N,m,offset=0):
    if m != 1:
        counter = [offset]
        for n in range(0,m-1):
            counter[0] = offset
            a(N,m,n,basis,N,m,counter)
        counter[0] = offset
        am(N,m,m-1,basis,N,m,counter)
    else:
        basis[offset,0] = int(N)
    #end

def basisOffsets(N,m):
    offsets = np.zeros( (N+2) ,dtype = np.int32)
    #set starting positions (counting from N,m-1 downwards) => results in rdm being block matrix of decreasing N_sub
    #set first one for avoiding exception in first nn value
    offsets[N] = 0
    #then following is offset[N-1] = offset[N] + #el(N,m-1) / however note offset[N]=0
    #offset[N-2] = offset[N-1] + #el(N-1,m_red)
    for i in reversed(range(-1,N)):
        offsets[i] = offsets[i+1] + dimOfBasis(i + 1, m)
    #note: offsets[N+1] = dim of basis
    return offsets

def fillReducedBasis(basis,N,m,offsets):
    for i in range(0,N+1):
        fillBasis(basis, i, m, offsets[i])
    #end
        
#filling arrays for l != m-1
def a(N,m,l,basis,Nsys,msys,counter):
    if m == msys-l:
        for n in range(0,N+1):
            nn = 0
            while nn < dimOfBasis(n,m-1):
                basis[counter[0]][l] = int(N-n)
                counter[0] += 1
                nn += 1
    else:
        for n in reversed(range(0,N+1)):
            a(N-n,m-1,l,basis,Nsys,msys,counter)
    #end

#filling arrays for l == m-1 (order is other way round)
def am(N,m,l,basis,Nsys,msys,counter):
    if m == msys:
        am(N,m-1,l,basis,Nsys,msys,counter)
    elif m == msys-l:
        for n in reversed(range(0,N+1)):
            basis[counter[0]][l] = int(N-n)
            counter[0] += 1
    else:
        for n in reversed(range(0,N+1)):
            am(N-n,m-1,l,basis,Nsys,msys,counter)
    #end

def basis2dict(basis,dim):
    #create an empty dictionary with states in occupation number repres. as tuples being the keys
    tup = tuple( tuple(el) for el in basis )
    dRet = dict.fromkeys(tup)
    #for correct correspondence go through the basis tuple and put in the vector-number corresponding to the given tuple
    for i in range(0,dim):
        dRet[tup[i]] = i
    return dRet

#note that all the elements here are sparse matrices! one has to use .toarray() to get them done correctly
def quadraticArray(sysVar):
    retArr = np.empty( (sysVar.m , sysVar.m), dtype = csr_matrix )
    #off diagonal
    for i in range(0, sysVar.m):
        for j in range(0, i):
            retArr[i,j] = getQuadratic(sysVar, i, j)
            retArr[j,i] = retArr[i,j].transpose()
    #diagonal terms
    for i in range(0, sysVar.m):
        retArr[i,i] = getQuadratic(sysVar, i, i)
    return retArr

#quadratic term in 2nd quantization for transition from m to l -> fills zero initialized matrix
#matrix for a_l^d a_m (r=row, c=column) is M[r][c] = SQRT(basis[r][l]*basis[c][m])
def getQuadratic(sysVar,l,m):
    data = np.zeros(0,dtype=np.float64)
    row = np.zeros(0,dtype=np.float64)
    col = np.zeros(0,dtype=np.float64)
    tmp = np.zeros( (sysVar.m), dtype = np.int)
    for el in sysVar.basis:
        if el[m] != 0:
            tmp = el.copy()
            tmp[m] -= 1
            tmp[l] += 1
            row = np.append(row,sysVar.basisDict[tuple(tmp)])
            col = np.append(col,sysVar.basisDict[tuple(el)])
            data = np.append(data,np.float64(sqrt(el[m])*sqrt(tmp[l])))
            
    retmat = coo_matrix((data, (row,col)), shape=(sysVar.dim , sysVar.dim) ,dtype = np.float64).tocsr()
    del row,col,data,tmp
    return retmat

#array elements are NO matrix! just numpy array!            
def quarticArray(sysVar):
    retArr = np.empty( (sysVar.m , sysVar.m, sysVar.m, sysVar.m), dtype = csr_matrix )
    # TODO: use transpose property
    for k in range(0, sysVar.m):
        for l in range(0, sysVar.m):
            for m in range(0, sysVar.m):
                for n in range(0, sysVar.m):
                        retArr[k,l,m,n] = getQuartic(sysVar, k, l, m, n)
    return retArr

def getQuartic(sysVar,k,l,m,n):
    if l != m:
        return (sysVar.operators[k,m] * sysVar.operators[l,n]).copy()
    else:
        return ((sysVar.operators[k,m] * sysVar.operators[l,n]) - sysVar.operators[k,n] ).copy()
    
def time_elapsed(t0,divider,decimals=0):
    t_el = tm.time() - t0
    if divider == 60:
        t_min = t_el // 60
        t_sec = t_el % 60
        return (str(int(t_min)) + "m " + str(int(t_sec)) + "s")
    else: 
        t_sec = t_el / divider
        return str(round(t_sec,decimals)) + "s"

#stores the matrix of dimension sysvar[2] in a file
def storeMatrix(mat,fil,absOnly=0):
    matDim = np.shape(mat)[0]
    if absOnly == 0:
        f = open(fil,'w')
        #assume dot + 3 letter ending e.g. .txt
        fname = fil[:-4]
        fend = fil[-4:]
        fimag = open(fname + '_im' + fend,'w')
        freal = open(fname + '_re' + fend,'w')
        for n in range(0,matDim):
            for nn in range(0,matDim-1):
                f.write(str(np.abs(mat[(n,nn)])) + ' ')
                fimag.write(str(np.imag(mat[(n,nn)])) + ' ')
                freal.write(str(np.real(mat[(n,nn)])) + ' ')
            f.write(str(np.abs(mat[(n,matDim-1)])) + "\n")
            fimag.write(str(np.imag(mat[(n,matDim-1)])) + "\n")
            freal.write(str(np.real(mat[(n,matDim-1)])) + "\n")
        f.close()
    else:
        f = open(fil,'w')
        #assume dot + 3 letter ending e.g. .txt
        for n in range(0,matDim):
            for nn in range(0,matDim-1):
                f.write(str(np.abs(mat[(n,nn)])) + ' ')
            f.write(str(np.abs(mat[(n,matDim-1)])) + "\n")
        f.close()   

#gives entropy -Tr(rho*log(rho) ) from density matrix (also works for reduced)
def densmat2ent(density_matrix):
    ent = 0
    for el in la.eigvalsh(density_matrix):
        if np.imag(el) != 0:
            print(np.imag(el))
        if np.real(el) > 0:
            ent -= np.real(el) * np.log(np.real(el))
        if np.real(el) < 0 and np.abs(el) > 1e-7:
            print('oh god, there is a negative eigenvalue smaller than 1e-7 !')
            print(el)
    return ent