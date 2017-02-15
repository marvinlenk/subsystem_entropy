import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

fwidth = 101
ford = 1

#this was for the m=4, special transition complex hamiltonian
def plotnL0(N,steps,full_periods,J):
    print("Plotting n1")
    
#     params={
#     'font.size': 12,
#     'mathtext.default' : 'rm' # see http://matplotlib.org/users/customizing.html
#     }
#     matplotlib.rcParams['agg.path.chunksize']=20000
#     matplotlib.rcParams.update(params)
    
    omegaT = J
    omegasqr = (np.sqrt(4* (J**2) + 1) / 2)
    delta_t = full_periods / steps
    
    nL0file = "./data/nL0.txt"
    series1 = np.loadtxt(nL0file)
    series1[:,0] = series1[:,0] * delta_t
    
    # with open("./data/n1.txt") as FILE:
    #     for LINE in FILE:
    #         data.append(LINE.strip().split()[0])
    
    analyticx = series1[:,0]
    analyticy = N * (np.cos(analyticx * np.pi * (omegaT/omegasqr)))**2 * ((4 * J**2 * (np.cos(analyticx * np.pi))**2 + 1) / (4 * J**2 + 1))
    #analyticinner = N * (np.cos(analyticx * np.pi * (omegaT/omegasqr)))**2
    analyticouter = N * ((4 * J**2 * (np.cos(analyticx * np.pi))**2 + 1) / (4 * J**2 + 1))
    
    plt.title('Occupation number n1 for N=' + str(N))
    plt.plot(analyticx,analyticy,label="Analytic", linewidth= 3.0, linestyle = '--')
    #plt.plot(analyticx,analyticinner,label="", linewidth= 1, color = 'r', LineStyle = ':')
    plt.plot(analyticx,analyticouter,label="", linewidth= 1, color = 'g', LineStyle = ':')
    plt.plot(series1[:,0],series1[:,1],label="Numerical", linewidth =1.0, color = 'r')
    
    #plt.colorbar(ax1)
    plt.grid(True)
    plt.xlabel('multiples of 2 pi')
    plt.ylabel('Occupation number')
    plt.legend(loc=4)
    #plt.set_xlim(A,B)
    plt.ylim(-1,N+1)
    plt.tight_layout()
    plt.savefig('./plots/nL0.png')
    plt.clf()
    
    deviation = series1
    deviation[:,1] = series1[:,1] - analyticy
    
    plt.title('Deviation from analytic solution')
    plt.ylabel('Deviation')
    plt.plot(deviation[:,0],deviation[:,1])
    plt.savefig('nL0dev.png')
    plt.clf()
    
    print("Plot of n1 done")
    
def plotData(sysVar):
    print("Plotting datapoints to pdf",end='')
    
    lsize = 13
    
    params={
        'font.size': 16,
        'mathtext.default' : 'rm' # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize']=20000
    plt.rcParams.update(params)
    
    pp = PdfPages('./plots/plots.pdf')
    
    occfile = "./data/occupation.txt"
    entfile = "./data/entropy.txt"
    normfile = "./data/norm.txt"
    energyfile = "./data/energy.txt"
    occ_array = np.loadtxt(occfile)
    step_array = occ_array[:,0]
    ent_array = np.loadtxt(entfile)
    norm_array = np.loadtxt(normfile)
    en_array = np.loadtxt(energyfile)
    en0 = en_array[0,1]
    en_array[:,1] -= en0
    #want deviation from 1
    norm_array[:,1] = 1 - norm_array[:,1]
    
    #### Plot Entropy
    if(sysVar.boolTotalEnt):
        plt.title('Entropy of whole system for N=%(N)i m=%(m)i' % {'N':sysVar.N, 'm':sysVar.m})
        plt.plot(step_array,ent_array[:,1], linewidth =0.6, color = 'r')
    
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('entropy')
        ###
        pp.savefig()
        plt.clf()
        print('.',end='')
    ###
    plt.title('Entropy of subsystem N=%(N)i m=%(m)i' % {'N':sysVar.N, 'm':sysVar.m})
    plt.plot(step_array,ent_array[:,2], linewidth =0.8, color = 'r')
    plt.grid()
    tavg = savgol_filter(ent_array[:,2],fwidth,ford)
    #plt.plot(step_array,tavg, linewidth =0.6, color = 'black')
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Occupation number of single levels')
    for i in range(0,sysVar.m):
        plt.plot(step_array,occ_array[:,i+1],label="n"+str(i), linewidth =0.5)
    
    plt.ylabel('Occupation number')
    plt.legend(loc='upper right',prop={'size':lsize})
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Occupation number of traced out System for N=%(N)i m=%(m)i' % {'N':sysVar.N, 'm':sysVar.m})
    for i in sysVar.kRed:
        plt.plot(step_array,occ_array[:,i+1],label="n"+str(i), linewidth =0.6)
    
    plt.ylabel('Occupation number')
    plt.legend(loc=4)
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Occupation number of leftover System for N=%(N)i m=%(m)i' % {'N':sysVar.N, 'm':sysVar.m})
    for i in np.arange(sysVar.m)[sysVar.mask]:
        plt.plot(step_array,occ_array[:,i+1],label="n"+str(i), linewidth =0.6)
    
    plt.ylabel('Occupation number')
    plt.legend(loc=4)
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Subsystem occupation number')
    tmp = np.zeros(len(step_array))
    for i in sysVar.kRed:
        tmp += occ_array[:,i+1]
    plt.plot(step_array,tmp,label="traced-out", linewidth =0.6, color = 'magenta')
    tavg = savgol_filter(tmp,fwidth,ford)
    #plt.plot(step_array,tavg, linewidth =0.6, color = 'black')
    avgt = 0
    vart = 0
    for i in range(0,np.shape(step_array[100:])[0]):
        avgt += tmp[i+100]
    avgt /= np.shape(step_array[100:])[0]
    
    for i in range(0,np.shape(step_array[100:])[0]):
        vart += (tmp[i+100] - avgt)**2
    vart /= np.shape(step_array[100:])[0]-1
    
    tmp.fill(0)
    for i in np.arange(sysVar.m)[sysVar.mask]:
        tmp += occ_array[:,i+1]
    plt.plot(step_array,tmp,label="leftover", linewidth =0.6, color = 'darkgreen')
    tavg = savgol_filter(tmp,fwidth,ford)
    #plt.plot(step_array,tavg, linewidth =0.6, color = 'black')
    avgl = 0
    varl = 0
    for i in range(0,np.shape(step_array[100:])[0]):
        avgl += tmp[i+100]
    avgl /= np.shape(step_array[100:])[0]
    
    for i in range(0,np.shape(step_array[100:])[0]):
        varl += (tmp[i+100] - avgl)**2
    varl /= np.shape(step_array[100:])[0]-1
    
    plt.ylabel('Occupation number')
    plt.legend(loc='center right', prop={'size':lsize})
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('relative Particle number fluctuations of system')
    tmp = np.zeros(len(step_array))
    for i in sysVar.kRed:
        tmp += occ_array[:,i+1]
    tmp2 = tmp**2
    stddev = np.sqrt(savgol_filter(tmp2,fwidth,ford) - savgol_filter(tmp,fwidth,ford)**2)
    tavg = savgol_filter(tmp,fwidth,ford)
    plt.plot(step_array,stddev/tavg, linewidth =0.6, color = 'black')
    plt.ylabel('fluctuation')
    plt.xlabel('J * t')
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Eigenenergies')
    engies = np.loadtxt('./eigvals.txt')
    plt.plot(engies,linestyle=':',linewidth=2)
    plt.ylabel('Energie')
    plt.xlabel('#')
    plt.grid(False)
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Energy of whole system for N=%(N)i m=%(m)i' % {'N':sysVar.N, 'm':sysVar.m}  + '  E(0) = ' +str(en0))
    plt.plot(step_array,en_array[:,1], linewidth =0.6)
    plt.ylabel('Energy - E(0)')
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('Norm of state deviation from 1')
    plt.plot(step_array,norm_array[:,1], "ro", ms=0.08, linewidth=1.1)
    
    plt.ylabel('norm deviation from 1')
    plt.grid(False)
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    plt.title('State Norm multiplied (deviation from 1)')
    plt.plot(step_array,norm_array[:,2]-1, linewidth =0.6, color = 'r')
    
    plt.ylabel('correction factor - 1')
    plt.grid()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='')
    ###
    pp.close()
    print('')
    print('traced out fluctuation:',np.sqrt(vart)/avgt,'mean value: ',avgt)
    print('leftover fluctuation:',np.sqrt(varl)/avgl,'mean value: ',avgl)
    print("done!")

def plotDensityMatrixAnimation(steps,delta_t,files,stepsize=1,red=0,framerate=30):
    if files%stepsize != 0:
        stepsize = int(files/100)
    if red == 0:
        rdstr = ''
    else:
        rdstr = 'red_'
        
    print("Plotting density matrix animation",end='')
    stor_step = steps / files
    fig = plt.figure(num=None, figsize=(30, 10), dpi=300)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    cax1 = fig.add_axes([0.06,0.1,0.02,0.8])
    cax2 = fig.add_axes([0.93,0.1,0.02,0.8])
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')    
    cmapVar.set_over(color='purple')    
    def iterate(n):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        plt.suptitle('t = %(time).2f' %{'time':n*stor_step*delta_t*stepsize})
        dabsfile = "./data/" +rdstr + "density/densmat" + str(int(n*stor_step)) + ".txt"
        dimagfile = "./data/" +rdstr + "density/densmat" + str(int(n*stor_step)) + "_im.txt"
        drealfile = "./data/" +rdstr + "density/densmat" + str(int(n*stor_step)) + "_re.txt"
        dabs = np.loadtxt(dabsfile)
        dimag = np.loadtxt(dimagfile)
        dreal = np.loadtxt(drealfile)
        ax1.set_xlabel('column')
        ax1.set_ylabel('row')
        ax1.set_title('absolute value')
        im = [ax1.imshow(dabs, cmap=cmapVar, interpolation='none',vmin=1e-16, vmax=1)]        
            
        ax2.set_title('real part')
        im.append(ax2.imshow(dreal, cmap=cmapVar, interpolation='none',vmin=1e-16, vmax=1))        
        if(n==0):
            fig.colorbar(im[1],cax=cax1)

        ax3.set_title('imaginary part')
        im.append(ax3.imshow(dimag, cmap=cmapVar, interpolation='none',vmin=-1, vmax=1))        
        if (n==0):
            fig.colorbar(im[2],cax=cax2)
        if n%( (files/stepsize) / 10) == 0:
            print('.',end='')
        return im
    
    ani = animation.FuncAnimation(fig, iterate, np.arange(0,files,stepsize))
    #ani.save('./plots/density.gif', writer='imagemagick') 
    ani.save('./plots/'+rdstr+'density.mp4',fps=framerate,extra_args=['-vcodec', 'libx264'],bitrate=-1)
    plt.close()
    print("done!")

def plotHamiltonian():     
    print("Plotting hamiltonian to pdf.",end='')
    pp = PdfPages('./plots/hamiltonian.pdf')
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    plt.title('absolute value of hamiltonian')
    dabs = np.loadtxt('./data/hamiltonian.txt')
    plt.xlabel('column')
    plt.ylabel('row')
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')   
    plt.imshow(dabs, cmap=cmapVar, interpolation='none',vmin=1e-20)   
    plt.colorbar()     
    pp.savefig()
    
    print('..',end='')
    
    plt.clf()
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    plt.title('absolute value of time evolution matrix')
    dabs = np.loadtxt('./data/evolutionmatrix.txt')
    plt.xlabel('column')
    plt.ylabel('row')
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')   
    plt.imshow(dabs, cmap=cmapVar, interpolation='none',vmin=1e-20)        
    plt.colorbar() 
    pp.savefig()
    
    pp.close()
    plt.close()
    print(" done!")

def plotmatrix2pdf(fPath):     
    print("Plotting matrix to pdf...",end='')
    fName = fPath.split('/')[-1:][0]
    pp = PdfPages('./plots/' + fName[:-4] + '.pdf')
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    plt.title('absolute value of matrix')
    dabs = np.loadtxt(fPath)
    plt.xlabel('column')
    plt.ylabel('row')
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')   
    plt.imshow(dabs, cmap=cmapVar, interpolation='none',vmin=1e-12)        
    pp.savefig()
    pp.close()
    plt.close()
    print("done!")
    