import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#searches for closest to value element in array
def find_nearest(array,value):
    i = (np.abs(array-value)).argmin()
    return int(i)

#This is a workaround until scipy fixes the issue
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def plotData(sysVar):
    print("Plotting datapoints to pdf",end='')
    
    avgstyle = 'dashed'
    avgsize = 0.6
    expectstyle = 'solid'
    expectsize = 1
    
    loavgind = 100 #index to start at when calculating average and stddev
    
    fwidth = sysVar.plotSavgolFrame
    ford = sysVar.plotSavgolOrder
    params={
        'legend.fontsize': sysVar.plotLegendSize,
        'font.size': sysVar.plotFontSize,
        'mathtext.default' : 'rm' # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize']=0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    pp = PdfPages('./plots/plots.pdf')
    
    occfile = './data/occupation.txt'
    occ_array = np.loadtxt(occfile)
    #multiply step array with time scale
    step_array = occ_array[:,0] * sysVar.plotTimeScale
    
    normfile = './data/norm.txt'
    norm_array = np.loadtxt(normfile)
    #want deviation from 1
    norm_array[:,1] = 1 - norm_array[:,1]
    
    entfile = './data/entropy.txt'
    ent_array = np.loadtxt(entfile)
    
    if sysVar.boolPlotEngy:
        engies = np.loadtxt('./data/hamiltonian_eigvals.txt')
    
    if sysVar.boolPlotDecomp:
        stfacts = np.loadtxt('./data/state.txt')
    
    if sysVar.boolTotalEnt:
        totentfile = './data/total_entropy.txt'
        totent_array = np.loadtxt(totentfile)
    
    if sysVar.boolTotalEnergy:
        energyfile = './data/energy.txt'
        en_array = np.loadtxt(energyfile)
        en0 = en_array[0,1]
        en_array[:,1] -= en0
        #en_micind = find_nearest(engies[:,1], en0)
        #print(' - |(E0 - Emicro)/E0|: %.0e - ' % (np.abs((en0 - engies[en_micind,1])/en0)), end='' )
    
    if sysVar.boolPlotDiagExp:
        microexpfile = './data/diagexpect.txt'
        microexp = np.loadtxt(microexpfile)
    
    #### Complete system Entropy
    if(sysVar.boolTotalEnt):
        plt.plot(totent_array[:,0]*sysVar.plotTimeScale,totent_array[:,1]*1e13, linewidth =0.6, color = 'r')
    
        plt.grid()
        plt.xlabel(r'$J\,t$')
        plt.ylabel(r'Total system entropy $/ 10^{-13}$')
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
    ### Subsystem Entropy
    plt.plot(step_array,ent_array[:,1], linewidth =0.8, color = 'r')
    plt.grid()
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(ent_array[:,1],fwidth,ford)
        plt.plot(step_array,tavg, linewidth = avgsize, linestyle=avgstyle, color = 'black')
    plt.xlabel(r'$J\,t$')
    plt.ylabel('Subsystem entropy')
    plt.tight_layout()
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ### Single-level occupation numbers
    for i in range(0,sysVar.m):
        plt.plot(step_array,occ_array[:,i+1],label=r'$n_'+str(i)+'$', linewidth =0.5)
        if sysVar.boolPlotAverages:
            tavg = savgol_filter(occ_array[:,i+1],fwidth,ford)
            plt.plot(step_array,tavg, linewidth =avgsize, linestyle=avgstyle, color = 'black')
        if sysVar.boolPlotDiagExp:
            plt.axhline(y=microexp[i,1], color='purple', linewidth = expectsize, linestyle = expectstyle)
    
    plt.ylabel(r'Occupation number')
    plt.xlabel(r'$J\,t$')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ### Traced out (bath) occupation numbers
    for i in sysVar.kRed:
        plt.plot(step_array,occ_array[:,i+1],label=r'$n_'+str(i)+'$', linewidth =0.6)
        if sysVar.boolPlotDiagExp:
            plt.axhline(y=microexp[i,1], color='purple', linewidth = expectsize, linestyle = expectstyle)
        if sysVar.boolPlotAverages:
            tavg = savgol_filter(occ_array[:,i+1],fwidth,ford)
            plt.plot(step_array,tavg, linewidth = avgsize, linestyle=avgstyle, color = 'black')
    
    plt.ylabel(r'Occupation number')
    plt.xlabel(r'$J\,t$')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ### Leftover (system) occupation numbers
    for i in np.arange(sysVar.m)[sysVar.mask]:
        plt.plot(step_array,occ_array[:,i+1],label=r'$n_'+str(i)+'$', linewidth =0.6)
        if sysVar.boolPlotDiagExp:
            plt.axhline(y=microexp[i,1], color='purple', linewidth = expectsize, linestyle = expectstyle)
        if sysVar.boolPlotAverages:
            tavg = savgol_filter(occ_array[:,i+1],fwidth,ford)
            plt.plot(step_array,tavg, linewidth = avgsize, linestyle=avgstyle, color = 'black')
    
    plt.ylabel(r'Occupation number')
    plt.xlabel(r'$J\,t$')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ### Subsystems occupation numbers
    #store fluctuations in a data
    fldat = open('./data/fluctuation.txt','w')
    fldat.write('N_tot: %i\n' % (sysVar.N))
    tmp = np.zeros(len(step_array))
    for i in sysVar.kRed:
        tmp += occ_array[:,i+1]
    plt.plot(step_array,tmp,label="bath", linewidth =0.8, color = 'magenta')
    
    tavg = savgol_filter(tmp,fwidth,ford)

    if sysVar.boolPlotAverages:
        plt.plot(step_array,tavg, linewidth = avgsize, linestyle=avgstyle, color = 'black')
    
    if sysVar.boolPlotDiagExp:
        mictmp = 0
        for i in sysVar.kRed:
            mictmp += microexp[i,1]
        plt.axhline(y=mictmp, color='purple', linewidth = expectsize, linestyle = expectstyle)
    
    avg = np.mean(tmp[loavgind:],dtype=np.float64)
    stddev = np.std(tmp[loavgind:],dtype=np.float64)
    fldat.write('bath_average: %.16e\n' % avg)
    fldat.write('bath_stddev: %.16e\n' % stddev)
    fldat.write('bath_rel._fluctuation: %.16e\n' % (stddev/avg))
    
    tmp.fill(0)
    for i in np.arange(sysVar.m)[sysVar.mask]:
        tmp += occ_array[:,i+1]
    plt.plot(step_array,tmp,label="system", linewidth =0.8, color = 'darkgreen')

    tavg = savgol_filter(tmp,fwidth,ford)
    if sysVar.boolPlotAverages:
        plt.plot(step_array,tavg, linewidth = avgsize, linestyle=avgstyle, color = 'black')
    
    if sysVar.boolPlotDiagExp:
        mictmp = 0
        for i in np.arange(sysVar.m)[sysVar.mask]:
            mictmp += microexp[i,1]
        plt.axhline(y=mictmp, color='purple', linewidth = expectsize, linestyle = expectstyle)
        
    avg = np.mean(tmp[loavgind:],dtype=np.float64)
    stddev = np.std(tmp[loavgind:],dtype=np.float64)
    fldat.write('system_average: %.16e\n' % avg)
    fldat.write('system_stddev: %.16e\n' % stddev)
    fldat.write('system_rel._fluctuation: %.16e\n' % (stddev/avg))
    
    for i in range(sysVar.m):
        avg = np.mean(occ_array[loavgind:,i+1],dtype=np.float64)
        stddev = np.std(occ_array[loavgind:,i+1],dtype=np.float64)
        fldat.write('n%i_average: %.16e\n' % (i,avg))
        fldat.write('n%i_stddev: %.16e\n' % (i,stddev))
        fldat.write('n%i_rel._fluctuation: %.16e\n' % (i,(stddev/avg)))
    
    avg = np.mean(ent_array[loavgind:,1],dtype=np.float64)
    stddev = np.std(ent_array[loavgind:,1],dtype=np.float64)
    fldat.write('ssentropy_average: %.16e\n' % avg)
    fldat.write('ssentropy_stddev: %.16e\n' % stddev)
    fldat.write('ssentropy_rel._fluctuation: %.16e\n' % (stddev/avg))
    
    
    
    fldat.close()
    
    plt.ylabel(r'Occupation number')
    plt.xlabel(r'$J\,t$')
    plt.legend(loc='center right')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ### Total system energy
    if sysVar.boolTotalEnergy:
        plt.title('$E_{tot}, \; E_0$ = %.2e' % en0)
        plt.plot(en_array[:,0]*sysVar.plotTimeScale,en_array[:,1]*1e10, linewidth =0.6)
        plt.ylabel(r'$E_{tot} - E_0 / 10^{-10}$')
        plt.xlabel(r'$J\,t$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
    ### Norm deviation
    plt.plot(step_array,norm_array[:,1], "ro", ms=0.5)
    plt.ylabel('norm deviation from 1')
    plt.xlabel(r'$J\,t$')
    plt.grid(False)
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ###
    plt.title('State Norm multiplied (deviation from 1)')
    plt.plot(step_array,norm_array[:,2]-1, linewidth =0.6, color = 'r')
    
    plt.ylabel('correction factor - 1')
    plt.xlabel(r'$J\,t$')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    ### Hamiltonian eigenvalues (Eigenenergies)
    if sysVar.boolPlotDecomp:
        plt.plot(engies[:,0],engies[:,1],linestyle='none',marker='o',ms=0.7,color='blue')
        plt.ylabel(r'Energy')
        plt.xlabel(r'\#')
        plt.grid(False)
        plt.xlim(xmin=-(len(engies[:,0]) * (5.0/100) ))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
    if sysVar.boolPlotDecomp:
        ### Hamiltonian eigenvalues (Eigenenergies) with decomposition
        fig, ax1 = plt.subplots()
        ax1.plot(engies[:,0],engies[:,1],linestyle='none',marker='o',ms=0.7,color='blue')
        ax1.set_ylabel(r'Energy')
        ax1.set_xlabel(r'\#')
        ax2 = ax1.twinx()
        ax2.bar(engies[:,0], engies[:,2], alpha=0.15,color='red',width=0.01,align='center')
        ax2.set_ylabel(r'$|c_n|^2$')
        plt.grid(False)
        ax1.set_xlim(xmin=-(len(engies[:,0]) * (5.0/100) ))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        ### Eigenvalue decomposition with energy x-axis
        plt.bar(engies[:,1], engies[:,2], alpha=0.5,color='red',width=0.01,align='center')
        plt.xlabel(r'Energy')
        plt.ylabel(r'$|c_n|^2$')
        plt.grid(False)
        plt.xlim(xmin=-( np.abs(engies[0,1] - engies[-1,1]) * (5.0/100) ))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        ### Eigenvalue decomposition en detail
        n_rows = 3 #abs**2, phase/2pi, energy on a range from 0 to 1 
        n_rows += 1 #spacer
        n_rows += sysVar.m #occupation numbers
        
        index = np.arange(sysVar.dim)
        bar_width = 1
        plt.xlim(0,sysVar.dim)
    
        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = np.array([0.0] * sysVar.dim)
        spacing = np.array([1] * sysVar.dim)
        enInt = np.abs(engies[-1,1] - engies[0,1])
        cmapVar = plt.cm.OrRd
        cmapVar.set_under(color='black')    
        plt.ylim(0,n_rows)
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar((engies[:,1]-engies[0,1])/enInt), linewidth=0.005, edgecolor='gray')
        y_offset = y_offset + spacing
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(engies[:,2]/np.amax(engies[:,2]) - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(engies[:,3] - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        
        plt.bar(index, spacing, bar_width, bottom=y_offset, color='white', linewidth=0)
        y_offset = y_offset + np.array([1] * sysVar.dim)
        
        for row in range(4, n_rows):
            plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(engies[:,row]/sysVar.N - 1e-16), linewidth=0.00, edgecolor='gray')
            y_offset = y_offset + spacing
        
        plt.ylabel("tba")
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        ### Occupation number basis decomposition en detail
        n_rows = 2 #abs**2, phase/2pi
        n_rows += 1 #spacer
        n_rows += sysVar.m #occupation numbers
        
        index = np.arange(sysVar.dim)
        bar_width = 1
        plt.xlim(0,sysVar.dim)
    
        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = np.array([0.0] * sysVar.dim)
        spacing = np.array([1] * sysVar.dim)
        cmapVar = plt.cm.OrRd
        cmapVar.set_under(color='black')    
        plt.ylim(0,n_rows)
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(stfacts[:,1]/np.amax(stfacts[:,1]) - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(stfacts[:,2] - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        
        plt.bar(index, spacing, bar_width, bottom=y_offset, color='white', linewidth=0)
        y_offset = y_offset + np.array([1] * sysVar.dim)
        
        for row in range(3, n_rows):
            plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(stfacts[:,row]/sysVar.N - 1e-16), linewidth=0.00, edgecolor='gray')
            y_offset = y_offset + spacing
        
        plt.ylabel("tba")
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
    ###
    pp.close()
    print(" done!")

def plotDensityMatrixAnimation(steps,delta_t,files,stepsize=1,red=0,framerate=30):
    if files%stepsize != 0:
        stepsize = int(files/100)
    if red == 0:
        rdstr = ''
        rdprstr = ''
    else:
        rdstr = 'red_'
        rdprstr = 'reduced-'
        
    print("Plotting "+redprstr+"density matrix animation",end='',flush=True)
    stor_step = steps / files
    fig = plt.figure(num=None, figsize=(30, 10), dpi=300)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    cax1 = fig.add_axes([0.06,0.1,0.02,0.8])
    cax2 = fig.add_axes([0.93,0.1,0.02,0.8])
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')      
    cmapVarIm = plt.cm.seismic  
    def iterate(n):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        plt.suptitle('t = %(time).2f' %{'time':n*stor_step*delta_t*stepsize})
        dabsfile = "./data/" +rdstr + "density/densmat" + str(int(n)) + ".txt"
        dimagfile = "./data/" +rdstr + "density/densmat" + str(int(n)) + "_im.txt"
        drealfile = "./data/" +rdstr + "density/densmat" + str(int(n)) + "_re.txt"
        dabs = np.loadtxt(dabsfile)
        dimag = np.loadtxt(dimagfile)
        dreal = np.loadtxt(drealfile)
        ax1.set_xlabel('column')
        ax1.set_ylabel('row')
        ax1.set_title('absolute value')
        im = [ax1.imshow(dabs, cmap=cmapVar, interpolation='none',vmin=1e-16)]        
            
        ax2.set_title('real part')
        im.append(ax2.imshow(dreal, cmap=cmapVar, interpolation='none',vmin=1e-16))        
        fig.colorbar(im[1],cax=cax1)

        ax3.set_title('imaginary part')
        im.append(ax3.imshow(dimag, cmap=cmapVarIm, interpolation='none'))        
        fig.colorbar(im[2],cax=cax2)
        if n%( (files/stepsize) / 10) == 0:
            print('.',end='',flush=True)
        return im
    
    ani = animation.FuncAnimation(fig, iterate, np.arange(0,files,stepsize))
    #ani.save('./plots/density.gif', writer='imagemagick') 
    ani.save('./plots/'+rdstr+'density.mp4',fps=framerate,extra_args=['-vcodec', 'libx264'],bitrate=-1)
    plt.close()
    print("done!")

def plotHamiltonian():     
    print("Plotting hamiltonian to pdf.",end='',flush=True)
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
    
    print('..',end='',flush=True)
    
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

def plotOccs(sysVar):     
    print("Plotting occupations to pdf.",end='',flush=True)
    pp = PdfPages('./plots/occs.pdf')
    params={
        'mathtext.default' : 'rm' # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize']=0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    for i in range(0,sysVar.m):
        plt.title(r'$n_'+str(i)+'$')
        dre = np.loadtxt('./data/occ'+str(i)+'_re.txt')
        plt.xlabel('column')
        plt.ylabel('row')
        cmapVar = plt.cm.seismic
        plt.imshow(dre, cmap=cmapVar, interpolation='none',vmin=-10,vmax=10)   
        cb=plt.colorbar()     
        pp.savefig()
        cb.remove()
        plt.clf

    pp.close()
    plt.close()
    print(" done!")
    
def plotTimescale(sysVar):     
    print("Plotting difference to mean.",end='',flush=True)
    pp = PdfPages('./plots/lndiff.pdf')
    params={
        'mathtext.default' : 'rm' # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize']=0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    
    loavgind = 100
    
    occfile = './data/occupation.txt'
    occ_array = np.loadtxt(occfile)
    #multiply step array with time scale
    step_array = occ_array[:,0] * sysVar.plotTimeScale
    
    entfile = './data/entropy.txt'
    ent_array = np.loadtxt(entfile)
    
    occavg = []
    for i in range(0,sysVar.m):
        occavg.append(np.mean(occ_array[loavgind:,i+1],dtype=np.float64))
    
    entavg = np.mean(ent_array[loavgind:,1],dtype=np.float64)
    
    odiff = []
    for i in range(0,sysVar.m):
        odiff.append(occ_array[:,i+1] - occavg[i])
    
    entdiff = ent_array[:,1] - entavg
    
    for i in range(0,sysVar.m):
        plt.ylabel(r'$\Delta n_%i$' % (i))
        plt.xlabel(r'$J\,t$')
        plt.plot(occ_array[:,0], odiff[i])
        pp.savefig()
        plt.clf()
        plt.ylabel(r'$\log | \Delta n_%i |$' % (i))
        plt.xlabel(r'$J\,t$')
        plt.plot(occ_array[:,0], np.log(np.abs(odiff[i])))
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        
    
    plt.ylabel(r'$\Delta S_{ss}$')
    plt.xlabel(r'$J\,t$')
    plt.plot(occ_array[:,0], entdiff[:])
    pp.savefig()
    plt.clf()
    plt.ylabel(r'$\log | \Delta S_{ss} |$')
    plt.xlabel(r'$J\,t$')
    plt.plot(occ_array[:,0], np.log(np.abs(entdiff[:])))
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    pp.close()
    print(" done!")
