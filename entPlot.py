import numpy as np
import os as os
import scipy.integrate as scint
import matplotlib as mpl
from scipy.optimize import least_squares
mpl.use('Agg')
from matplotlib.pyplot import cm , step
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy.integrate import cumtrapz

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
    
    loavgpercent = sysVar.plotLoAvgPerc #percentage of time evolution to start averaging
    loavgind = int(loavgpercent*sysVar.dataPoints) #index to start at when calculating average and stddev
    loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale),2)
    
    if sysVar.boolPlotAverages:
        print(' with averaging from Jt=%.2f' % loavgtime,end='')
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
    
    if sysVar.boolPlotOffDiag:
        offdiagfile = './data/offdiagonal.txt'
        offdiag = np.loadtxt(offdiagfile)
    
    if sysVar.boolPlotOffDiagDens:
        offdiagdensfile = './data/offdiagonaldens.txt'
        offdiagdens = np.loadtxt(offdiagdensfile)
    
    if sysVar.boolPlotGreen:
        greenfile = './data/green.txt'
        greendat = np.loadtxt(greenfile)
    
    def complete_system_enttropy():
        return 0    
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
    
    def subsystem_entropy():
        return 0    
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
    '''
    ###FFT
    print('')
    fourier = np.fft.rfft(ent_array[loavgind:,1])
    print(fourier[0].real)
    freq = np.fft.rfftfreq(np.shape(ent_array[loavgind:,1])[-1], d=step_array[1])
    plt.plot(freq[1:],np.abs(fourier[1:]))
    print('')
    plt.ylabel(r'$A_{\omega}$')
    plt.xlabel(r'$\omega$')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    '''
    
    def single_level_occ():
        return 0    
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
    '''
    ###FFT
    print('')
    for i in range(0,sysVar.m):
        plt.xlim(xmax=30)
        #GK = -i(2n-1)
        fourier = (rfft(occ_array[loavgind:,i+1],norm='ortho'))*2 -1
        print(fourier[0].real)
        freq = rfftfreq(np.shape(occ_array[loavgind:,i+1])[-1], d=step_array[1])
        plt.plot(freq,fourier.real,linewidth = 0.05)
        plt.plot(freq,fourier.imag,linewidth = 0.05)
        plt.ylabel(r'$G^K_{\omega}$')
        plt.xlabel(r'$\omega$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
    print('.',end='',flush=True)
    '''
    def bath_occ():
        return 0
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
    
    def system_occ():
        return 0    
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
    
    def subsystem_occupation():
        return 0    
    ### Subsystems occupation numbers
    #store fluctuations in a data
    fldat = open('./data/fluctuation.txt','w')
    fldat.write('N_tot: %i\n' % (sysVar.N))
    tmp = np.zeros(len(step_array))
    for i in sysVar.kRed:
        tmp += occ_array[:,i+1]
    plt.plot(step_array,tmp,label="bath", linewidth =0.8, color = 'magenta')

    if sysVar.boolPlotAverages:
        tavg = savgol_filter(tmp,fwidth,ford)
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

    if sysVar.boolPlotAverages:
        tavg = savgol_filter(tmp,fwidth,ford)
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
    
    def occ_distribution():
        return 0    
    #occupation number in levels against level index
    occavg = np.loadtxt('./data/fluctuation.txt', usecols=(1,))
    plt.xlim(-0.1,sysVar.m-0.9)
    for l in range(0,sysVar.m):
        plt.errorbar(l,occavg[int(7 + 3*l)]/sysVar.N,xerr=None,yerr=occavg[int(8 + 3*l)]/sysVar.N,marker='o',color=cm.Set1(0))
    plt.ylabel(r'Relative level occupation')
    plt.xlabel(r'Level index')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    
    def sum_offdiagonals():
        return 0    
    #sum of off diagonal elements in energy eigenbasis
    if sysVar.boolPlotOffDiag:
        for i in range(0,sysVar.m):
            plt.plot(step_array,offdiag[:,i+1],label=r'$n_'+str(i)+'$', linewidth =0.5)
        plt.ylabel(r'Sum of off diagonals')
        plt.xlabel(r'$J\,t$')
        plt.legend(loc='upper right')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        
        
        dt = offdiag[1,0]-offdiag[0,0]
        nrm = offdiag[:,0]/dt
        nrm[1:] = 1/nrm[1:]
        for i in range(0,sysVar.m):
            ###### only sum (subsystem-thermalization)
            plt.ylabel('Sum of off diagonals in $n^{%i}$' % (i))
            # start at 10% of the whole x-axis
            lox = (offdiag[-1,0]-offdiag[0,0])/10 + offdiag[0,0]
            hiy = offdiag[ int(len(offdiag[:,0])/10) ,0] * 1.1
            plt.plot(offdiag[:,0],offdiag[:,i+1],linewidth = 0.5)
            plt.xlim(xmin=lox)
            plt.ylim(ymax=hiy)
            plt.grid()
            plt.tight_layout()
            ###inlay with the whole deal
            a = plt.axes([0.62, 0.6, 0.28, 0.28])
            a.plot(offdiag[:,0],offdiag[:,i+1],linewidth = 0.8)
            a.set_xticks([])
            a.set_yticks([])
            ###
            pp.savefig()
            plt.clf()
            
            plt.ylabel('Sum of off diagonals in $n^{%i}$' % (i))
            plt.semilogy(offdiag[:,0],np.abs(offdiag[:,i+1]),linewidth = 0.5)
            plt.xlim(xmin=lox)
            plt.ylim(ymin=1e-2)
            plt.grid()
            plt.tight_layout()
            ###inlay with the whole deal
            a = plt.axes([0.62, 0.6, 0.28, 0.28])
            a.semilogy(offdiag[:,0],offdiag[:,i+1],linewidth = 0.8)
            a.set_ylim(ymin=1e-2)
            a.set_xticks([])
            a.set_yticks([])
            ###
            pp.savefig()
            plt.clf()
            
            ###### average (eigenstate-thermalization)
            f, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)
            tmp = cumtrapz(offdiag[:,i+1],offdiag[:,0],initial=offdiag[0,i+1])
            tmp = np.multiply(tmp,nrm)
            f.text(0.03, 0.5, 'Average of summed off diagonals in $n^{%i}$' % (i), ha='center', va='center', rotation='vertical')
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax1.plot(offdiag[:,0],tmp,linewidth = 0.5)
            ax1.grid()
            
            ax2.loglog(offdiag[:,0],np.abs(tmp),linewidth = 0.5)
            ax2.set_ylim(bottom=1e-4)
            ax2.grid()
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.12)
            ###
            pp.savefig()
            plt.clf()
        
        print('.',end='',flush=True)
    
    def sum_offdiagonalsdens():
        return 0
    if sysVar.boolPlotOffDiagDens:
        plt.plot(step_array,offdiagdens[:,1], linewidth =0.5)
        plt.ylabel(r'Sum of off diagonals (red. dens. mat.)')
        plt.xlabel(r'$J\,t$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
    
    def total_energy():
        return 0    
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
        
    def norm_deviation():
        return 0    
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
    
    def eigenvalues():
        return 0    
    ### Hamiltonian eigenvalues (Eigenenergies)
    if sysVar.boolPlotEngy:
        linearize = False
        if linearize:
            tap = []
            lal = -1
            for e in engies[:,1]:
                if lal == -1:
                    tap.append(e)
                    lal += 1
                elif np.abs(e - tap[lal]) > 1:
                    lal += 1
                    tap.append(e)
            plt.plot(tap,linestyle='none',marker='o',ms=0.5,color='blue')
        else:
            plt.plot(engies[:,0],engies[:,1],linestyle='none',marker='o',ms=0.5,color='blue')
        
        plt.ylabel(r'E/J')
        plt.xlabel(r'\#')
        plt.grid(False)
        plt.xlim(xmin=-(len(engies[:,0]) * (2.0/100) ))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
    
    def density_of_states():
        return 0    
    ### DOS
    if sysVar.boolPlotDOS:
        dos = np.zeros(sysVar.dim)
        window = 50
        iw = window
        for i in range(iw,sysVar.dim-iw):
            dos[i] = (window)*2/(engies[i+iw,1] - engies[i-iw,1])
        dos /= (sysVar.dim-iw)
        print(scint.simps(dos[iw:], engies[iw:,1]))
        plt.plot(engies[:,1],dos,lw=0.005)
        plt.ylabel(r'DOS')
        plt.xlabel(r'E')
        plt.grid(False)
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
    
    def greensfunction():
        return 0    
    ### Greensfunction
    if sysVar.boolPlotGreen:
        gd = greendat
        '''
        gd = np.zeros((np.shape(greendat)[0]*2,np.shape(greendat)[1]))
        gd[int(np.shape(greendat)[0]/2):-int(np.shape(greendat)[0]/2 + np.shape(greendat)[0]%2)] = greendat[:,:].copy()
        '''
        spec = []
        discpoints = len(gd[:,0])
        print('')
        for i in range(0,sysVar.m):
            plt.title(r'two time Green function of level $%i$' % (i))
            ind = 2*i + 1
            plt.plot(greendat[:,0]*sysVar.plotTimeScale,greendat[:,ind],lw=0.1,color='red',label='real')
            plt.plot(greendat[:,0]*sysVar.plotTimeScale,greendat[:,ind+1],lw=0.1,color='blue',label='imaginary')
            #plt.xlim(xmax=10)
            plt.ylabel(r'$G^R(\tau)$')
            plt.xlabel(r'$J\,\tau$')
            plt.legend(loc='lower right')
            plt.grid()
            plt.tight_layout()
            ###
            pp.savefig()
            plt.clf()
            ###
            plt.title(r'Spectral function of level $%i$' % (i))
            green_ret = gd[:,ind] + 1j * gd[:,ind+1]
            green_ret_freq = fft(np.hanning(len(green_ret))*green_ret,norm='ortho')
            spec_tmp = np.abs(-2*fftshift(green_ret_freq.imag))[::-1]
            if i == 0:
                samp_spacing = sysVar.deltaT * (sysVar.steps / sysVar.dataPoints) * sysVar.plotTimeScale
                hlpfrq = fftshift(fftfreq(len(spec_tmp)))*(2*np.pi)/samp_spacing
            ### !!! normalize by hand! this might be strange but is necessary here
            spec_tmp /= (np.trapz(spec_tmp,x=hlpfrq)/(2*np.pi))
            if i == 0:
                spec_total = spec_tmp[:]
                # scale on x-axis is frequency
            else:
                spec_total += spec_tmp
            spec.append(spec_tmp)
            print(i,np.trapz(spec_tmp, x = hlpfrq)/(2*np.pi))
            #exit()
            plt.plot(hlpfrq,spec_tmp,color = 'red',lw=0.1)
            plt.minorticks_on()
            plt.ylabel(r'$A$')
            plt.xlabel(r'$\omega / J$')

            plt.grid()
            plt.grid(which='minor', color='blue', linestyle='dotted', lw=0.2)
            plt.tight_layout()
            ###
            pp.savefig()
            plt.clf()
        plt.title(r'Spectral function')
        plt.plot(hlpfrq,spec_total,color = 'red',lw=0.1)
        plt.ylabel(r'$A$')
        plt.xlabel(r'$\omega / J$')
        #plt.xlim([-100,100])
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        plt.plot(hlpfrq,np.abs(spec_total),color = 'red',lw=0.1)
        plt.ylabel(r'$|A|$')
        plt.xlabel(r'$\omega / J$')
        #plt.xlim([-100,100])
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
          
        print('.',end='',flush=True)
        
        print()
        weights = np.zeros(len(spec))
        for s in range(0,len(spec)):
            print(np.average(hlpfrq,weights=spec[s]), np.average(hlpfrq,weights=np.abs(spec[s])))
            weights[s] = np.abs(np.average(hlpfrq,weights=np.abs(spec[s])))
        print('')
        '''
        # the integrated version
        def occno(spec,freq,temp,mu):
            rt = []
            for i in range(0,len(freq)):
                rt.append(spec[i]/(np.exp((freq[i]-mu)/temp)-1.0))
            return np.trapz(np.array(rt), x=freq)
        '''
        # the averaged version
        def occno(freq,temp,mu):
                return (1/(np.exp((freq-mu)/temp)-1.0))
            
        def bestatd(args):
            temp = args[0]
            mu = args[1]
            ret =[]
            for i in range(0,sysVar.m):
                ret.append(occno(weights[i],temp,mu) - occavg[int(7 + 3*i)])
            return np.array(ret)
        
        def bestat(args):
            temp = args[0]
            mu = args[1]
            ret =[]
            for i in range(0,sysVar.m):
                ret.append(occno(weights[i],temp,mu))
            return np.array(ret)
        
        strt = np.array([10,-0.1])
        bnds = np.array([[0.0001,-500],[1000,weights[0]]])
        rgs = least_squares(bestatd,x0=strt,bounds=bnds,loss='soft_l1')
        print(rgs)
        print(rgs.x)
        print(bestat(rgs.x))
        a = []
        for i in range(0,sysVar.m):
            a.append(occavg[int(7+3*i)])
        print(a)
        
        #occupation number in levels against renormalized energy
        plt.title('Bose-Einstein distribution fit')
        ws = np.sort(weights)
        lo = ws[0]-ws[0]/100
        hi = ws[-1]+ws[-1]/100
        plt.xlim(lo,hi)
        xvals = np.linspace(lo, hi, 1e3)
        yvals = occno(xvals, rgs.x[0], rgs.x[1]) / sysVar.N
        for l in range(0,sysVar.m):
            plt.errorbar(weights[l],occavg[int(7 + 3*l)]/sysVar.N,xerr=None,yerr=occavg[int(8 + 3*l)]/sysVar.N,marker='o',color=cm.Set1(0))
        plt.plot(xvals,yvals,color='blue',lw=0.4)
        plt.ylabel(r'Relative level occupation')
        plt.xlabel(r'energy')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        
    if sysVar.boolPlotDecomp:
        def eigendecomposition():
            return 0    
        ### Hamiltonian eigenvalues (Eigenenergies) with decomposition
        fig, ax1 = plt.subplots()
        ax1.plot(engies[:,0],engies[:,1],linestyle='none',marker='o',ms=0.7,color='blue')
        ax1.set_ylabel(r'Energy')
        ax1.set_xlabel(r'\#')
        ax2 = ax1.twinx()
        ax2.bar(engies[:,0], engies[:,2], alpha=0.8,color='red',width=0.03,align='center')
        ax2.set_ylabel(r'$|c_n|^2$')
        plt.grid(False)
        ax1.set_xlim(xmin=-(len(engies[:,0]) * (5.0/100) ))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        ### Eigenvalue decomposition with energy x-axis
        plt.bar(engies[:,1], engies[:,2], alpha=0.8,color='red',width=0.03, align='center')
        plt.xlabel(r'Energy')
        plt.ylabel(r'$|c_n|^2$')
        plt.grid(False)
        plt.xlim(xmin=-( np.abs(engies[0,1] - engies[-1,1]) * (5.0/100) ))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        # omit this in general
        '''
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
        #energy
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar((engies[:,1]-engies[0,1])/enInt), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        #abs squared
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(engies[:,2]/np.amax(engies[:,2]) - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        #phase / 2pi
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(engies[:,3] - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        
        plt.bar(index, spacing, bar_width, bottom=y_offset, color='white', linewidth=0)
        y_offset = y_offset + np.array([1] * sysVar.dim)
        
        #expectation values
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
        # abs squared
        plt.bar(index, spacing , bar_width, bottom=y_offset, color=cmapVar(stfacts[:,1]/np.amax(stfacts[:,1]) - 1e-16), linewidth=0.00, edgecolor='gray')
        y_offset = y_offset + spacing
        # phase / 2pi
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
        '''
    def densmat_spectral():
        return 0    
    ####### Density matrix in spectral repesentation
    if sysVar.boolPlotSpectralDensity:
        ###
        plt.title('Density matrix spectral repres. abs')
        dabs = np.loadtxt('./data/spectral/dm.txt')
        cmapVar = plt.cm.Reds
        cmapVar.set_under(color='black') 
        plt.imshow(dabs, cmap=cmapVar, interpolation='none',vmin=1e-16)
        plt.colorbar()
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
        plt.imshow(dre, cmap=cmapVar, interpolation='none', vmin=-sysVar.N, vmax=sysVar.N)   
        cb=plt.colorbar()     
        pp.savefig()
        cb.remove()
        plt.clf
        print('.',end='',flush=True)
    
    # now without diagonals and abs only
    for i in range(0,sysVar.m):
        plt.title(r'$n_'+str(i)+'$')
        dre = np.loadtxt('./data/occ'+str(i)+'_re.txt')
        np.fill_diagonal(dre, 0)
        plt.xlabel('column')
        plt.ylabel('row')
        cmapVar = plt.cm.Reds
        cmapVar.set_under(color='black') 
        plt.imshow(np.abs(dre), cmap=cmapVar, interpolation='none',vmin=1e-6)   
        cb=plt.colorbar()     
        pp.savefig()
        cb.remove()
        plt.clf
        print('.',end='',flush=True)
    
    pp.close()
    plt.close()
    print(" done!")

def plotOffDiagSingles(sysVar):
    print("Plotting off-diagonal singles.", end='', flush=True)
    params={
        'legend.fontsize': sysVar.plotLegendSize,
        'font.size': sysVar.plotFontSize,
        'mathtext.default' : 'rm' # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize']=0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    pp = PdfPages('./plots/offdiagsingles.pdf')
    
    singlesdat = np.loadtxt('./data/offdiagsingle.txt')
    singlesinfo = np.loadtxt('./data/offdiagsingleinfo.txt')
    
    dt = singlesdat[1,0]-singlesdat[0,0]
    nrm = singlesdat[:,0]/dt
    nrm[1:] = 1/nrm[1:]
    
    '''
    for i in range(0,sysVar.m):
        for j in range(0,sysVar.occEnSingle):
            infoind = 1+4*j+2 #so we start at the first energy
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
            f.suptitle(r'$n_{%i} \; E_1=%.2e \; E_2=%.2e$' % (i, singlesinfo[i,infoind], singlesinfo[i,infoind+1]))
            ind = 1+2*j+(i*sysVar.occEnSingle*2)
            comp = singlesdat[:,ind] + 1j*singlesdat[:,ind+1]
            ax1.set_ylabel(r'$|A_{n,m}|$')
            ax1.plot(singlesdat[:,0], np.abs(comp), linewidth = 0.5)
            tmp = cumtrapz(comp,singlesdat[:,0]/dt,initial=comp[0])
            tmp = np.multiply(tmp,nrm)
            ax2.set_ylabel(r'average $|A_{n,m}|$')
            ax2.plot(singlesdat[:,0], np.abs(tmp), linewidth = 0.5)
            ax3.set_ylabel(r'arg$/\pi$')
            plt.xlabel(r'$J\,t$')
            ax3.plot(singlesdat[:,0], np.angle(comp)/(np.pi), linewidth = 0.5)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, left=0.1)
            pp.savefig(f)
            f.clf()
            # do the double log plot
            de = np.abs(singlesinfo[i,infoind] - singlesinfo[i,infoind+1])
            linar = np.zeros(len(singlesdat[:,0]), dtype=np.float64)
            linar[0] = 0
            linar[1:] = 2/(singlesdat[1:,0] * de)
            plt.xlabel(r'$J\,t$')
            plt.ylabel(r'relative average $|A_{n,m}|$')
            plt.loglog(singlesdat[1:,0], np.abs(tmp/np.abs(comp[0]))[1:], singlesdat[1:,0], linar[1:], lw=0.5)
            pp.savefig()
            plt.clf()
        print('.',end='',flush=True)
    '''
    for i in range(0,sysVar.m):
        for j in range(0,sysVar.occEnSingle):
            infoind = 1+4*j+2 #so we start at the first energy
            # fetch the exponents. if abs(ordr)==1 set to zero for more readability
            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
            ordr1 = int(np.log10(np.abs(singlesinfo[i,infoind])))
            if ordr1 == 1 or ordr1 == -1:
                ordr1 = 0
            ordr2 = int(np.log10(np.abs(singlesinfo[i,infoind+1])))
            if ordr2 == 1 or ordr2 == -1:
                ordr2 = 0
            if ordr1 == 0 and ordr2 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \; E_m=%.2f$' % (i, singlesinfo[i,infoind], singlesinfo[i,infoind+1]))
            elif ordr1 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \; E_m=%.2f \cdot 10^{%i}$' % (i, singlesinfo[i,infoind], singlesinfo[i,infoind+1]/(10**ordr2), ordr2))
            elif ordr2 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \cdot 10^{%i} \; E_m=%.2f$' % (i, singlesinfo[i,infoind]/(10**ordr1), ordr1, singlesinfo[i,infoind+1]))
            else:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \cdot 10^{%i} \; E_m=%.2f \cdot 10^{%i}$' % (i, singlesinfo[i,infoind]/(10**ordr1), ordr1, singlesinfo[i,infoind+1]/(10**ordr2), ordr2))
            #
            ind = 1+2*j+(i*sysVar.occEnSingle*2)
            comp = singlesdat[:,ind] + 1j*singlesdat[:,ind+1]
            
            # order of magnitude of the deviation
            if not (np.abs(np.abs(comp[0]) - np.abs(comp[-1])) == 0):
                ordr = int(np.log10(np.abs(np.abs(comp[0]) - np.abs(comp[-1])))) - 1
            else:
                ordr = 0
            ax1.set_ylabel(r'$|n_{n,m}^{%i}(t)| - |n_{n,m}^{%i}(0)| / 10^{%i}$' % (i,i,ordr))
            ax1.plot(singlesdat[:,0], (np.abs(comp)-np.abs(comp[0]))/(10**ordr), linewidth = 0.5)
            tmp = cumtrapz(comp,singlesdat[:,0]/dt,initial=comp[0])
            tmp = np.multiply(tmp,nrm)
            
            # order of magnitude of the average
            if not (np.abs(tmp[1]) == 0):
                ordr = int(np.log10(np.abs(tmp[1]))) - 1
            else:
                ordr = 0
            ax2.set_ylabel(r'$|\overline{n}_{n,m}^{%i}| / 10^{%i}$' % (i, ordr))
            ax2.plot(singlesdat[:,0], np.abs(tmp)/(10**ordr), linewidth = 0.5)
            ax2.set_xlabel(r'$J\,t$')
            plt.tight_layout()
            pp.savefig(f)
            f.clf()
            plt.close()
            # do the double log plot
            de = np.abs(singlesinfo[i,infoind] - singlesinfo[i,infoind+1])
            linar = np.zeros(len(singlesdat[:,0]), dtype=np.float64)
            linar[0] = 0
            linar[1:] = 2/(singlesdat[1:,0] * de)
            plt.xlabel(r'$J\,t$')
            plt.ylabel(r'relative average $|n_{n,m}^{%i}|$' % (i))
            plt.loglog(singlesdat[1:,0], np.abs(tmp/np.abs(comp[0]))[1:], singlesdat[1:,0], linar[1:], lw=0.5)
            pp.savefig()
            plt.clf()
            plt.close()
        print('.',end='',flush=True)
    diagdat = np.loadtxt('./data/diagsingles.txt')
    
    if os.path.isfile('./data/energy.txt') and os.path.isfile('./data/hamiltonian_eigvals.txt'):
    ### look for energy - this works because the energies are sorted
        engy = np.loadtxt('./data/energy.txt')
        eigengy = np.loadtxt('./data/hamiltonian_eigvals.txt')
        diff = 0
        for l in range(0,sysVar.dim):
            if np.abs(eigengy[l,1] - engy[0,1]) > diff and l != 0:
                eind = l-1
                break
            else:
                diff = np.abs(eigengy[l,1] - engy[0,1])
        if eind < 15:
            loran = 0
        else:
            loran = eind-15
            
    for i in range(0,sysVar.m):
        if os.path.isfile('./data/energy.txt') and os.path.isfile('./data/hamiltonian_eigvals.txt'):
            plt.title(r'Diagonal weighted elements of $n_{%i}$ in spectral decomp.' % (i))
            lo = np.int32(sysVar.dim * i)
            hi = np.int32(lo + sysVar.dim)
            plt.ylabel(r'$|n%i_{E}|$' % (i))
            plt.xlabel(r'$E / J$')
            #plt.plot(diagdat[lo:hi,1], diagdat[lo:hi,2],linestyle='none',marker='o',ms=0.5)
            plt.plot(diagdat[lo+loran:hi,1][:30], diagdat[lo+loran:hi,2][:30] ,marker='o',ms=2) 
            plt.axvline(x=engy[0,1], linewidth=0.8, color='red')
            
            ###inlay
            a = plt.axes([0.18, 0.6, 0.28, 0.28])
            a.plot(diagdat[lo:hi-300,1], diagdat[lo:hi-300,2],marker='o',ms=0.6, ls = 'none')
            a.set_xticks([])
            a.set_yticks([])
                
            pp.savefig()
            plt.clf()
            if os.path.isfile('./data/occ'+str(i)+'_re.txt'):
                occmat = np.loadtxt('./data/occ'+str(i)+'_re.txt')
                diags = np.zeros(sysVar.dim)
                
                ### large plot
                plt.title(r'Diagonal elements of $n_{%i}$ in spectral decomposition' % (i))
                plt.ylabel(r'$|n%i_{E}|$' % (i))
                plt.xlabel(r'$E / J$')
                for el in range(0,sysVar.dim):
                    diags[el] = occmat[el,el]
                
                plt.plot(diagdat[lo+loran:hi,1][:30], diags[loran:][:30] ,marker='o',ms=2) 
                plt.axvline(x=engy[0,1], linewidth=0.8, color='red')
                ### inlay
                a = plt.axes([0.18, 0.6, 0.28, 0.28])
                a.plot(diagdat[lo:hi-50,1], diags[:-50] ,marker='o',ms=0.5, ls='none')
                a.set_xticks([])
                a.set_yticks([])
                pp.savefig()
                plt.clf()
        else:
            plt.title(r'Diagonal weighted elements of $n_{%i}$ in spectral decomp.' % (i))
            lo = np.int32(sysVar.dim * i)
            hi = np.int32(lo + sysVar.dim)
            plt.ylabel(r'$|n%i_{E}|$' % (i))
            plt.xlabel(r'$E / J$')
            #plt.plot(diagdat[lo:hi,1], diagdat[lo:hi,2],linestyle='none',marker='o',ms=0.5)
            plt.plot(diagdat[lo:hi-200,1], diagdat[lo:hi-200,2],marker='o',ms=0.6, ls = 'none')
            plt.tight_layout()
            pp.savefig()
            plt.clf()
           
    print('.',end='',flush=True)
    pp.close()
    plt.close()
    print(' done!')

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
    
    ### get the characteristic energy difference of the system
    if sysVar.boolEngyStore:
        engys = np.loadtxt('./data/hamiltonian_eigvals.txt')
        enscale = np.abs(engys[0,1] - engys[-1,1])/sysVar.dim
        del engys
    
    loavgpercent = sysVar.plotLoAvgPerc #percentage of time evolution to start averaging
    loavgind = int(loavgpercent*sysVar.dataPoints) #index to start at when calculating average and stddev
    loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale),2)
    
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
        plt.plot(occ_array[:,0], odiff[i], lw=0.5)
        if sysVar.boolEngyStore:
            plt.axvline(enscale,color='red',lw=0.5)
        pp.savefig()
        plt.clf()
        plt.ylabel(r'$| \Delta n_%i |$' % (i))
        plt.xlabel(r'$J\,t$')
        plt.ylim(ymin=1e-3)
        plt.semilogy(occ_array[:,0], np.abs(odiff[i]),lw=0.5)
        if sysVar.boolEngyStore:
            plt.axvline(enscale,color='red',lw=0.5)
        plt.tight_layout()
        pp.savefig()
        plt.clf()
        print('.',end='',flush=True)
        
    
    plt.ylabel(r'$\Delta S_{ss}$')
    plt.xlabel(r'$J\,t$')
    plt.plot(occ_array[:,0], entdiff[:],lw=0.5)
    if sysVar.boolEngyStore:
            plt.axvline(enscale,color='red',lw=0.5)
    pp.savefig()
    plt.clf()
    plt.ylabel(r'$| \Delta S_{ss} |$')
    plt.xlabel(r'$J\,t$')
    plt.ylim(ymin=1e-3)
    plt.semilogy(occ_array[:,0], np.abs(entdiff[:]),lw=0.5)
    if sysVar.boolEngyStore:
            plt.axvline(enscale,color='red',lw=0.5)
    plt.tight_layout()
    pp.savefig()
    plt.clf()
    print('.',end='',flush=True)
    pp.close()
    print(" done!")
