import numpy as np
import dft
import os as os
import scipy.integrate as scint
import matplotlib as mpl

mpl.use('Agg')
from matplotlib.pyplot import cm, step
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz


# searches for closest to value element in array
def find_nearest(array, value):
    i = (np.abs(array - value)).argmin()
    return int(i)


# This is a workaround until scipy fixes the issue
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# noinspection PyStringFormat
def plotData(sysVar):
    print("Plotting datapoints to pdf", end='')

    avgstyle = 'dashed'
    avgsize = 0.6
    expectstyle = 'solid'
    expectsize = 1

    loavgpercent = sysVar.plotLoAvgPerc  # percentage of time evolution to start averaging
    loavgind = int(loavgpercent * sysVar.dataPoints)  # index to start at when calculating average and stddev
    loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale), 2)

    if sysVar.boolPlotAverages:
        print(' with averaging from Jt=%.2f' % loavgtime, end='')
    fwidth = sysVar.plotSavgolFrame
    ford = sysVar.plotSavgolOrder
    params = {
        'legend.fontsize': sysVar.plotLegendSize,
        'font.size': sysVar.plotFontSize,
        'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize'] = 0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    pp = PdfPages(sysVar.dataFolder + 'plots.pdf')

    if sysVar.boolOccupations:
        occfile = './data/occupation.dat'
        occ_array = np.loadtxt(occfile)

    normfile = './data/norm.dat'
    norm_array = np.loadtxt(normfile)
    # want deviation from 1
    norm_array[:, 1] = 1 - norm_array[:, 1]

    if sysVar.boolReducedEntropy:
        entfile = './data/entropy.dat'
        ent_array = np.loadtxt(entfile)

    if sysVar.boolPlotEngy:
        engies = np.loadtxt('./data/hamiltonian_eigvals.dat')

    if sysVar.boolPlotDecomp:
        stfacts = np.loadtxt('./data/state.dat')

    if sysVar.boolTotalEntropy:
        totentfile = './data/total_entropy.dat'
        totent_array = np.loadtxt(totentfile)

    if sysVar.boolTotalEnergy:
        energyfile = './data/total_energy.dat'
        en_array = np.loadtxt(energyfile)
        en0 = en_array[0, 1]
        en_array[:, 1] -= en0
        # en_micind = find_nearest(engies[:,1], en0)
        # print(' - |(E0 - Emicro)/E0|: %.0e - ' % (np.abs((en0 - engies[en_micind,1])/en0)), end='' )

    if sysVar.boolReducedEnergy:
        redEnergyfile = './data/reduced_energy.dat'
        redEnergy = np.loadtxt(redEnergyfile)

    if sysVar.boolPlotOccEnDiag:
        occendiag = []
        occendiagweighted = []
        for i in range(sysVar.m):
            occendiag.append(np.loadtxt('./data/occ_energybasis_diagonals_%i.dat' % i))
            occendiagweighted.append(np.loadtxt('./data/occ_energybasis_diagonals_weighted_%i.dat' % i))

    if sysVar.boolPlotOccEnDiagExp:
        microexpfile = './data/occ_energybasis_diagonals_expectation.dat'
        microexp = np.loadtxt(microexpfile)

    if sysVar.boolPlotOffDiagOcc:
        offdiagoccfile = './data/offdiagocc.dat'
        offdiagocc = np.loadtxt(offdiagoccfile)

    if sysVar.boolPlotOffDiagDens:
        offdiagdensfile = './data/offdiagdens.dat'
        offdiagdens = np.loadtxt(offdiagdensfile)

    if sysVar.boolPlotOffDiagDensRed:
        offdiagdensredfile = './data/offdiagdensred.dat'
        offdiagdensred = np.loadtxt(offdiagdensredfile)

    if sysVar.boolPlotGreen:
        greenfile = './data/' + [s for s in os.listdir('./data/') if 'green' in s][0]
        greendat = np.loadtxt(greenfile)

    def complete_system_enttropy():
        return 0
        # ### Complete system Entropy

    if sysVar.boolTotalEntropy:
        plt.plot(totent_array[:, 0] * sysVar.plotTimeScale, totent_array[:, 1] * 1e13, linewidth=0.6, color='r')

        plt.grid()
        plt.xlabel(r'$J\,t$')
        plt.ylabel(r'Total system entropy $/ 10^{-13}$')
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    def subsystem_entropy():
        return 0
        # ## Subsystem Entropy
    if sysVar.boolReducedEntropy:
        step_array = ent_array[:, 0] * sysVar.plotTimeScale
        plt.plot(step_array, ent_array[:, 1], linewidth=0.8, color='r')
        plt.grid()
        if sysVar.boolPlotAverages:
            tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
            plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
        plt.xlabel(r'$J\,t$')
        plt.ylabel('Subsystem entropy')
        plt.tight_layout()
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

        # Subsystem entropy with inlay
        max_time = step_array[-1]
        max_ind = int(max_time / step_array[-1] * len(step_array))

        avg = np.mean(ent_array[loavgind:, 1])
        plt.plot(step_array[:], ent_array[:, 1], linewidth=0.3, color='r')
        if sysVar.boolPlotAverages:
            tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
            plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
        plt.xlabel(r'$J\,t$')
        plt.ylabel(r'Subsystem entropy $S\textsubscript{sys}$')
        a = plt.axes([.5, .3, .4, .4])
        plt.semilogy(step_array[:max_ind], np.abs(avg - ent_array[:max_ind, 1]), linewidth=0.3, color='r')
        plt.ylabel(r'$|\,\overline{S}\textsubscript{sys} - S\textsubscript{sys}(t)|$')
        plt.yticks([])
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

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
        # ## Single-level occupation numbers

    if sysVar.boolOccupations:
        step_array = occ_array[:, 0] * sysVar.plotTimeScale
        for i in range(sysVar.m):
            plt.plot(step_array, occ_array[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.5)
            if sysVar.boolPlotAverages:
                tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
                plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
            if sysVar.boolPlotOccEnDiagExp:
                plt.axhline(y=microexp[i, 1], color='purple', linewidth=expectsize, linestyle=expectstyle)

        plt.ylabel(r'Occupation number')
        plt.xlabel(r'$J\,t$')
        plt.legend(loc='upper right')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)
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

        # ## Traced out (bath) occupation numbers
        for i in sysVar.kRed:
            plt.plot(step_array, occ_array[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.6)
            if sysVar.boolPlotOccEnDiagExp:
                plt.axhline(y=microexp[i, 1], color='purple', linewidth=expectsize, linestyle=expectstyle)
            if sysVar.boolPlotAverages:
                tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
                plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

        plt.ylabel(r'Occupation number')
        plt.xlabel(r'$J\,t$')
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

        def system_occ():
            return 0
            # ## Leftover (system) occupation numbers

        for i in np.arange(sysVar.m)[sysVar.mask]:
            plt.plot(step_array, occ_array[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.6)
            if sysVar.boolPlotOccEnDiagExp:
                plt.axhline(y=microexp[i, 1], color='purple', linewidth=expectsize, linestyle=expectstyle)
            if sysVar.boolPlotAverages:
                tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
                plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

        plt.ylabel(r'Occupation number')
        plt.xlabel(r'$J\,t$')
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

        def subsystem_occupation():
            return 0
            # ## Subsystems occupation numbers

        # store fluctuations in a data
        fldat = open('./data/fluctuation.dat', 'w')
        fldat.write('N_tot: %i\n' % (sysVar.N))
        tmp = np.zeros(len(step_array))
        for i in sysVar.kRed:
            tmp += occ_array[:, i + 1]
        plt.plot(step_array, tmp, label="bath", linewidth=0.8, color='magenta')

        if sysVar.boolPlotAverages:
            tavg = savgol_filter(tmp, fwidth, ford)
            plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

        if sysVar.boolPlotOccEnDiagExp:
            mictmp = 0
            for i in sysVar.kRed:
                mictmp += microexp[i, 1]
            plt.axhline(y=mictmp, color='purple', linewidth=expectsize, linestyle=expectstyle)

        avg = np.mean(tmp[loavgind:], dtype=np.float64)
        stddev = np.std(tmp[loavgind:], dtype=np.float64)
        fldat.write('bath_average: %.16e\n' % avg)
        fldat.write('bath_stddev: %.16e\n' % stddev)
        # noinspection PyStringFormat
        fldat.write("bath_rel._fluctuation: %.16e\n" % (stddev / avg))

        tmp.fill(0)
        for i in np.arange(sysVar.m)[sysVar.mask]:
            tmp += occ_array[:, i + 1]
        plt.plot(step_array, tmp, label="system", linewidth=0.8, color='darkgreen')

        if sysVar.boolPlotAverages:
            tavg = savgol_filter(tmp, fwidth, ford)
            plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

        if sysVar.boolPlotOccEnDiagExp:
            mictmp = 0
            for i in np.arange(sysVar.m)[sysVar.mask]:
                mictmp += microexp[i, 1]
            plt.axhline(y=mictmp, color='purple', linewidth=expectsize, linestyle=expectstyle)

        avg = np.mean(tmp[loavgind:], dtype=np.float64)
        stddev = np.std(tmp[loavgind:], dtype=np.float64)
        fldat.write('system_average: %.16e\n' % avg)
        fldat.write('system_stddev: %.16e\n' % stddev)
        fldat.write('system_rel._fluctuation: %.16e\n' % (stddev / avg))

        for i in range(sysVar.m):
            avg = np.mean(occ_array[loavgind:, i + 1], dtype=np.float64)
            stddev = np.std(occ_array[loavgind:, i + 1], dtype=np.float64)
            fldat.write('n%i_average: %.16e\n' % (i, avg))
            fldat.write('n%i_stddev: %.16e\n' % (i, stddev))
            fldat.write('n%i_rel._fluctuation: %.16e\n' % (i, (stddev / avg)))

        if sysVar.boolReducedEntropy:
            avg = np.mean(ent_array[loavgind:, 1], dtype=np.float64)
            stddev = np.std(ent_array[loavgind:, 1], dtype=np.float64)
            fldat.write('ssentropy_average: %.16e\n' % avg)
            fldat.write('ssentropy_stddev: %.16e\n' % stddev)
            fldat.write('ssentropy_rel._fluctuation: %.16e\n' % (stddev / avg))

        fldat.close()

        plt.ylabel(r'Occupation number')
        plt.xlabel(r'$J\,t$')
        plt.legend(loc='center right')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

        def occ_distribution():
            return 0
            # occupation number in levels against level index

        occavg = np.loadtxt('./data/fluctuation.dat', usecols=(1,))
        plt.xlim(-0.1, sysVar.m - 0.9)
        for l in range(sysVar.m):
            plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                         marker='o', color=cm.Set1(0))
        plt.ylabel(r'Relative level occupation')
        plt.xlabel(r'Level index')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    def sum_offdiagonals():
        return 0
        # sum of off diagonal elements in energy eigenbasis

    if sysVar.boolPlotOffDiagOcc:
        for i in range(sysVar.m):
            plt.plot(step_array, offdiagocc[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.5)
        plt.ylabel(r'Sum of off diagonals')
        plt.xlabel(r'$J\,t$')
        plt.legend(loc='upper right')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()

        dt = offdiagocc[1, 0] - offdiagocc[0, 0]
        nrm = offdiagocc[:, 0] / dt
        nrm[1:] = 1 / nrm[1:]
        for i in range(sysVar.m):
            # ##### only sum (subsystem-thermalization)
            plt.ylabel('Sum of off diagonals in $n^{%i}$' % (i))
            # start at 10% of the whole x-axis
            lox = (offdiagocc[-1, 0] - offdiagocc[0, 0]) / 10 + offdiagocc[0, 0]
            hiy = offdiagocc[int(len(offdiagocc[:, 0]) / 10), 0] * 1.1
            plt.plot(offdiagocc[:, 0], offdiagocc[:, i + 1], linewidth=0.5)
            plt.xlim(xmin=lox)
            plt.ylim(ymax=hiy)
            plt.grid()
            plt.tight_layout()
            # ##inlay with the whole deal
            a = plt.axes([0.62, 0.6, 0.28, 0.28])
            a.plot(offdiagocc[:, 0], offdiagocc[:, i + 1], linewidth=0.8)
            a.set_xticks([])
            a.set_yticks([])
            ###
            pp.savefig()
            plt.clf()

            plt.ylabel('Sum of off diagonals in $n^{%i}$' % (i))
            plt.semilogy(offdiagocc[:, 0], np.abs(offdiagocc[:, i + 1]), linewidth=0.5)
            plt.xlim(xmin=lox)
            plt.ylim(ymin=1e-2)
            plt.grid()
            plt.tight_layout()
            # ##inlay with the whole deal
            a = plt.axes([0.62, 0.6, 0.28, 0.28])
            a.semilogy(offdiagocc[:, 0], offdiagocc[:, i + 1], linewidth=0.8)
            a.set_ylim(ymin=1e-2)
            a.set_xticks([])
            a.set_yticks([])
            ###
            pp.savefig()
            plt.clf()

            # ##### average (eigenstate-thermalization)
            f, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)
            tmp = cumtrapz(offdiagocc[:, i + 1], offdiagocc[:, 0], initial=offdiagocc[0, i + 1])
            tmp = np.multiply(tmp, nrm)
            f.text(0.03, 0.5, 'Average of summed off diagonals in $n^{%i}$' % i, ha='center', va='center',
                   rotation='vertical')
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax1.plot(offdiagocc[:, 0], tmp, linewidth=0.5)
            ax1.grid()

            ax2.loglog(offdiagocc[:, 0], np.abs(tmp), linewidth=0.5)
            ax2.set_ylim(bottom=1e-4)
            ax2.grid()

            plt.tight_layout()
            plt.subplots_adjust(left=0.12)
            ###
            pp.savefig()
            plt.clf()

        print('.', end='', flush=True)

    def sum_offdiagonalsdens():
        return 0

    if sysVar.boolPlotOffDiagDens:
        plt.plot(step_array, offdiagdens[:, 1], linewidth=0.5)
        plt.ylabel(r'Sum of off diagonals (dens. mat.)')
        plt.xlabel(r'$J\,t$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()

    def sum_offdiagonalsdensred():
        return 0

    if sysVar.boolPlotOffDiagDensRed:
        plt.plot(step_array, offdiagdensred[:, 1], linewidth=0.5)
        plt.ylabel(r'Sum of off diagonals (red. dens. mat.)')
        plt.xlabel(r'$J\,t$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()

    def occupation_energybasis_diagonals():
        return 0

    if sysVar.boolPlotOccEnDiag:
        for i in range(sysVar.m):
            plt.plot(occendiag[i][:, 1], occendiag[i][:, 2], marker='o', markersize=0.5, linestyle=None)
            plt.ylabel(r'matrix element of $n_%i$' % i)
            plt.xlabel(r'$E / J$')
            plt.grid()
            plt.tight_layout()
            ###
            pp.savefig()
            plt.clf()
            ###
            plt.plot(occendiagweighted[i][:, 1], occendiagweighted[i][:, 2], marker='o', markersize=0.5, linestyle=None)
            plt.ylabel(r'weighted matrix element of $n_%i$' % i)
            plt.xlabel(r'$E / J$')
            plt.grid()
            plt.tight_layout()
            ###
            pp.savefig()
            plt.clf()

    def total_energy():
        return 0
        # ## Total system energy

    if sysVar.boolTotalEnergy:
        plt.title('$E_{tot}, \; E_0$ = %.2f' % en0)
        plt.plot(en_array[:, 0] * sysVar.plotTimeScale, en_array[:, 1] * 1e10, linewidth=0.6)
        plt.ylabel(r'$E_{tot} - E_0 / 10^{-10}$')
        plt.xlabel(r'$J\,t$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    def reduced_energy():
        return 0
        # ## Total system energy

    if sysVar.boolReducedEnergy:
        plt.plot(redEnergy[:, 0] * sysVar.plotTimeScale, redEnergy[:, 1], linewidth=0.6)
        plt.ylabel(r'$E_{sys}$')
        plt.xlabel(r'$J\,t$')
        plt.grid()
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    def norm_deviation():
        return 0
        # ## Norm deviation

    step_array = norm_array[:, 0] * sysVar.plotTimeScale
    plt.plot(step_array, norm_array[:, 1], "ro", ms=0.5)
    plt.ylabel('norm deviation from 1')
    plt.xlabel(r'$J\,t$')
    plt.grid(False)
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.', end='', flush=True)
    ###
    plt.title('State Norm multiplied (deviation from 1)')
    plt.plot(step_array, norm_array[:, 2] - 1, linewidth=0.6, color='r')

    plt.ylabel('correction factor - 1')
    plt.xlabel(r'$J\,t$')
    plt.grid()
    plt.tight_layout()
    ###
    pp.savefig()
    plt.clf()
    print('.', end='', flush=True)

    def eigenvalues():
        return 0
        # ## Hamiltonian eigenvalues (Eigenenergies)

    if sysVar.boolPlotEngy:
        linearize = False
        if linearize:
            tap = []
            lal = -1
            for e in engies[:, 1]:
                if lal == -1:
                    tap.append(e)
                    lal += 1
                elif np.abs(e - tap[lal]) > 1:
                    lal += 1
                    tap.append(e)
            plt.plot(tap, linestyle='none', marker='o', ms=0.5, color='blue')
        else:
            plt.plot(engies[:, 0], engies[:, 1], linestyle='none', marker='o', ms=0.5, color='blue')

        plt.ylabel(r'Energy / J')
        plt.xlabel(r'Eigenvalue Index')
        plt.grid(False)
        plt.xlim(xmin=-(len(engies[:, 0]) * (2.0 / 100)))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    def density_of_states():
        return 0
        # ## DOS

    if sysVar.boolPlotDOS:
        dos = np.zeros(sysVar.dim)
        window = 50
        iw = window
        for i in range(iw, sysVar.dim - iw):
            dos[i] = (window) * 2 / (engies[i + iw, 1] - engies[i - iw, 1])
        dos /= (sysVar.dim - iw)
        print(scint.simps(dos[iw:], engies[iw:, 1]))
        plt.plot(engies[:, 1], dos, lw=0.005)
        plt.ylabel(r'Density of states')
        plt.xlabel(r'Energy / J')
        plt.grid(False)
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    def green_function():
        return 0
        # ## Green function

    if sysVar.boolPlotGreen:

        if False:
            if not(os.path.isfile('./data/spectral_frequency_trace.dat') and
                       os.path.isfile('./data/statistical_frequency_trace.dat')):
                spectral_time = []
                statistical_time = []
                spectral_frequency = []
                statistical_frequency = []
                sample_frequency = 1.0 / ((greendat[-1, 0] - greendat[0, 0]) / (len(greendat[:, 0]) - 1))
                for i in range(sysVar.m):
                    ind = 1 + i * 4
                    greater = (greendat[:, ind] + 1j * greendat[:, ind + 1])  # greater green function
                    lesser = (greendat[:, ind + 2] + 1j * greendat[:, ind + 3])  # lesser green function
                    spectral_time.append(1j * (greater - lesser))  # spectral function in time space
                    statistical_time.append((greater + lesser) / 2)  # statistical function in time space
                    # spectral function in frequency space
                    spectral_frequency.append(dft.rearrange(dft.dft(spectral_time[i], sample_frequency)))
                    # statistical function in frequency space
                    statistical_frequency.append(dft.rearrange(dft.dft(statistical_time[i], sample_frequency)))

                spectral_frequency_trace = spectral_frequency[0]
                statistical_frequency_trace = statistical_frequency[0]
                for i in range(1, sysVar.m):
                    spectral_frequency_trace[:, 1] = spectral_frequency_trace[:, 1] + spectral_frequency[i][:, 1]
                    statistical_frequency_trace[:, 1] = statistical_frequency_trace[:, 1] + statistical_frequency[i][:, 1]

                np.savetxt('./data/spectral_frequency_trace.dat',
                           np.column_stack((spectral_frequency_trace[:, 0].real, spectral_frequency_trace[:, 1].real,
                                            spectral_frequency_trace[:, 1].imag)))
                np.savetxt('./data/statistical_frequency_trace.dat',
                           np.column_stack((statistical_frequency_trace[:, 0].real, statistical_frequency_trace[:, 1].real,
                                            statistical_frequency_trace[:, 1].imag)))
            else:
                spectral_frequency_trace_tmp = np.loadtxt('./data/spectral_frequency_trace.dat')
                statistical_frequency_trace_tmp = np.loadtxt('./data/statistical_frequency_trace.dat')
                spectral_frequency_trace = np.column_stack((
                    spectral_frequency_trace_tmp[:, 0],
                    (spectral_frequency_trace_tmp[:, 1] + 1j*spectral_frequency_trace_tmp[:, 2])
                ))
                statistical_frequency_trace = np.column_stack((
                    statistical_frequency_trace_tmp[:, 0],
                    (statistical_frequency_trace_tmp[:, 1] + 1j*statistical_frequency_trace_tmp[:, 2])
                ))

            fig = plt.figure()

            ax1 = fig.add_subplot(221)
            ax1.set_xlim(xmin=-10, xmax=100)
            ax1.set_ylabel(r'Re $A(\omega)$')
            ax1.set_xlabel(r'$\omega$')
            ax1.plot(spectral_frequency_trace[:, 0].real, spectral_frequency_trace[:, 1].real)

            ax2 = fig.add_subplot(222)
            ax2.set_xlim(xmin=-10, xmax=100)
            ax2.set_ylabel(r'Im $A(\omega)$')
            ax2.set_xlabel(r'$\omega$')
            ax2.plot(spectral_frequency_trace[:, 0].real, spectral_frequency_trace[:, 1].imag)

            ax3 = fig.add_subplot(223)
            ax3.set_xlim(xmin=-10, xmax=100)
            ax3.set_ylabel(r'Re $F(\omega)$')
            ax3.set_xlabel(r'$\omega$')
            ax3.plot(statistical_frequency_trace[:, 0].real, statistical_frequency_trace[:, 1].real)

            ax4 = fig.add_subplot(224)
            ax4.set_xlim(xmin=-10, xmax=100)
            ax4.set_ylabel(r'Im $F(\omega)$')
            ax4.set_xlabel(r'$\omega$')
            ax4.plot(statistical_frequency_trace[:, 0].real, statistical_frequency_trace[:, 1].imag)

            plt.tight_layout()
            pp.savefig()
            plt.clf()

            # ## ##

            fig = plt.figure()

            ax1 = fig.add_subplot(121)
            ax1.set_xlim(xmin=-10, xmax=80)
            ax1.set_ylabel(r'$|A(\omega)|$')
            ax1.set_xlabel(r'$\omega$')
            ax1.plot(spectral_frequency_trace[:, 0].real, np.abs(spectral_frequency_trace[:, 1]))

            ax2 = fig.add_subplot(122)
            ax2.set_xlim(xmin=-10, xmax=80)
            ax2.set_ylabel(r'$|F(\omega)|$')
            ax2.set_xlabel(r'$\omega$')
            ax2.plot(statistical_frequency_trace[:, 0].real, np.abs(statistical_frequency_trace[:, 1]))

            plt.tight_layout()
            pp.savefig()
            plt.clf()
            print('.', end='', flush=True)

    if sysVar.boolPlotDecomp:
        def eigendecomposition():
            return 0
            ### Hamiltonian eigenvalues (Eigenenergies) with decomposition

        fig, ax1 = plt.subplots()
        energy_markersize = 0.7
        energy_barsize = 0.06
        if sysVar.dim != 1:
            energy_markersize *= (2.0 / np.log10(sysVar.dim))
            energy_barsize *= (4.0 / np.log10(sysVar.dim))
        ax1.plot(engies[:, 0], engies[:, 1], linestyle='none', marker='o', ms=energy_markersize, color='blue')
        ax1.set_ylabel(r'Energy / J')
        ax1.set_xlabel(r'Eigenvalue index n')
        ax2 = ax1.twinx()
        ax2.bar(engies[:, 0], engies[:, 2], alpha=0.8, color='red', width=0.03, align='center')
        ax2.set_ylabel(r'$|c_n|^2$')
        plt.grid(False)
        ax1.set_xlim(xmin=-(len(engies[:, 0]) * (5.0 / 100)))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)
        ### Eigenvalue decomposition with energy x-axis
        plt.bar(engies[:, 1], engies[:, 2], alpha=0.8, color='red', width=energy_barsize, align='center')
        plt.xlabel(r'Energy / J')
        plt.ylabel(r'$|c_E|^2$')
        plt.grid(False)
        plt.xlim(xmin=-(np.abs(engies[0, 1] - engies[-1, 1]) * (5.0 / 100)))
        plt.tight_layout()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)
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
        dabs = np.loadtxt('./data/spectral/dm.dat')
        cmapVar = plt.cm.Reds
        cmapVar.set_under(color='black')
        plt.imshow(dabs, cmap=cmapVar, interpolation='none', vmin=1e-16)
        plt.colorbar()
        ###
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)
    ###
    pp.close()
    print(" done!")


def plotDensityMatrixAnimation(steps, delta_t, files, stepsize=1, red=0, framerate=30):
    if files % stepsize != 0:
        stepsize = int(files / 100)
    if red == 0:
        rdstr = ''
        rdprstr = ''
    else:
        rdstr = 'red_'
        rdprstr = 'reduced-'

    print("Plotting " + rdprstr + "density matrix animation", end='', flush=True)
    stor_step = steps / files
    fig = plt.figure(num=None, figsize=(30, 10), dpi=300)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    cax1 = fig.add_axes([0.06, 0.1, 0.02, 0.8])
    cax2 = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')
    cmapVarIm = plt.cm.seismic

    def iterate(n):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        plt.suptitle('t = %(time).2f' % {'time': n * stor_step * delta_t * stepsize})
        dabsfile = "./data/" + rdstr + "density/densmat" + str(int(n)) + ".dat"
        dimagfile = "./data/" + rdstr + "density/densmat" + str(int(n)) + "_im.dat"
        drealfile = "./data/" + rdstr + "density/densmat" + str(int(n)) + "_re.dat"
        dabs = np.loadtxt(dabsfile)
        dimag = np.loadtxt(dimagfile)
        dreal = np.loadtxt(drealfile)
        ax1.set_xlabel('column')
        ax1.set_ylabel('row')
        ax1.set_title('absolute value')
        im = [ax1.imshow(dabs, cmap=cmapVar, interpolation='none', vmin=1e-16)]

        ax2.set_title('real part')
        im.append(ax2.imshow(dreal, cmap=cmapVar, interpolation='none', vmin=1e-16))
        fig.colorbar(im[1], cax=cax1)

        ax3.set_title('imaginary part')
        im.append(ax3.imshow(dimag, cmap=cmapVarIm, interpolation='none'))
        fig.colorbar(im[2], cax=cax2)
        if n % ((files / stepsize) / 10) == 0:
            print('.', end='', flush=True)
        return im

    ani = animation.FuncAnimation(fig, iterate, np.arange(files, stepsize))
    # ani.save(sysVar.dataFolder + 'density.gif', writer='imagemagick')
    ani.save(sysVar.dataFolder + rdstr + 'density.mp4', fps=framerate, extra_args=['-vcodec', 'libx264'], bitrate=-1)
    plt.close()
    print("done!")


def plotHamiltonian():
    print("Plotting hamiltonian to pdf.", end='', flush=True)
    pp = PdfPages(sysVar.dataFolder + 'hamiltonian.pdf')
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    plt.title('absolute value of hamiltonian')
    dabs = np.loadtxt('./data/hamiltonian.dat')
    plt.xlabel('column')
    plt.ylabel('row')
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')
    plt.imshow(dabs, cmap=cmapVar, interpolation='none', vmin=1e-20)
    plt.colorbar()
    pp.savefig()

    print('..', end='', flush=True)

    plt.clf()
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    plt.title('absolute value of time evolution matrix')
    dabs = np.loadtxt('./data/evolutionmatrix.dat')
    plt.xlabel('column')
    plt.ylabel('row')
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')
    plt.imshow(dabs, cmap=cmapVar, interpolation='none', vmin=1e-20)
    plt.colorbar()
    pp.savefig()

    pp.close()
    plt.close()
    print(" done!")


def plotOccs(sysVar):
    print("Plotting occupations to pdf.", end='', flush=True)
    pp = PdfPages(sysVar.dataFolder + 'occs.pdf')
    params = {
        'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize'] = 0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    for i in range(sysVar.m):
        plt.title(r'$n_' + str(i) + '$')
        dre = np.loadtxt('./data/occ_energybasis_%i_re.dat' % i)
        plt.xlabel('column')
        plt.ylabel('row')
        cmapVar = plt.cm.seismic
        plt.imshow(dre, cmap=cmapVar, interpolation='none', vmin=-sysVar.N, vmax=sysVar.N)
        cb = plt.colorbar()
        pp.savefig()
        cb.remove()
        plt.clf
        print('.', end='', flush=True)

    # now without diagonals and abs only
    for i in range(sysVar.m):
        plt.title(r'$n_' + str(i) + '$')
        dre = np.loadtxt('./data/occ_energybasis_%i_re.dat' % i)
        np.fill_diagonal(dre, 0)
        plt.xlabel('column')
        plt.ylabel('row')
        cmapVar = plt.cm.Reds
        cmapVar.set_under(color='black')
        plt.imshow(np.abs(dre), cmap=cmapVar, interpolation='none', vmin=1e-6)
        cb = plt.colorbar()
        pp.savefig()
        cb.remove()
        plt.clf
        print('.', end='', flush=True)

    pp.close()
    plt.close()
    print(" done!")


def plotOffDiagOccSingles(sysVar):
    print("Plotting off-diagonal singles.", end='', flush=True)
    params = {
        'legend.fontsize': sysVar.plotLegendSize,
        'font.size': sysVar.plotFontSize,
        'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize'] = 0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    pp = PdfPages(sysVar.dataFolder + 'offdiagsingles.pdf')

    singlesdat = np.loadtxt('./data/offdiagsingle.dat')
    singlesinfo = np.loadtxt('./data/offdiagsingleinfo.dat')

    dt = singlesdat[1, 0] - singlesdat[0, 0]
    nrm = singlesdat[:, 0] / dt
    nrm[1:] = 1 / nrm[1:]

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
    for i in range(sysVar.m):
        for j in range(sysVar.occEnSingle):
            infoind = 1 + 4 * j + 2  # so we start at the first energy
            # fetch the exponents. if abs(ordr)==1 set to zero for more readability
            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
            ordr1 = int(np.log10(np.abs(singlesinfo[i, infoind])))
            if ordr1 == 1 or ordr1 == -1:
                ordr1 = 0
            ordr2 = int(np.log10(np.abs(singlesinfo[i, infoind + 1])))
            if ordr2 == 1 or ordr2 == -1:
                ordr2 = 0
            if ordr1 == 0 and ordr2 == 0:
                f.suptitle(
                    r'$n_{%i} \quad E_n=%.2f \; E_m=%.2f$' % (i, singlesinfo[i, infoind], singlesinfo[i, infoind + 1]))
            elif ordr1 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \; E_m=%.2f \cdot 10^{%i}$' % (
                    i, singlesinfo[i, infoind], singlesinfo[i, infoind + 1] / (10 ** ordr2), ordr2))
            elif ordr2 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \cdot 10^{%i} \; E_m=%.2f$' % (
                    i, singlesinfo[i, infoind] / (10 ** ordr1), ordr1, singlesinfo[i, infoind + 1]))
            else:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \cdot 10^{%i} \; E_m=%.2f \cdot 10^{%i}$' % (
                    i, singlesinfo[i, infoind] / (10 ** ordr1), ordr1, singlesinfo[i, infoind + 1] / (10 ** ordr2),
                    ordr2))
            #
            ind = 1 + 2 * j + (i * sysVar.occEnSingle * 2)
            comp = singlesdat[:, ind] + 1j * singlesdat[:, ind + 1]

            # order of magnitude of the deviation
            if not (np.abs(np.abs(comp[0]) - np.abs(comp[-1])) == 0):
                ordr = int(np.log10(np.abs(np.abs(comp[0]) - np.abs(comp[-1])))) - 1
            else:
                ordr = 0
            ax1.set_ylabel(r'$|n_{n,m}^{%i}(t)| - |n_{n,m}^{%i}(0)| / 10^{%i}$' % (i, i, ordr))
            ax1.plot(singlesdat[:, 0], (np.abs(comp) - np.abs(comp[0])) / (10 ** ordr), linewidth=0.5)
            tmp = cumtrapz(comp, singlesdat[:, 0] / dt, initial=comp[0])
            tmp = np.multiply(tmp, nrm)

            # order of magnitude of the average
            if not (np.abs(tmp[1]) == 0):
                ordr = int(np.log10(np.abs(tmp[1]))) - 1
            else:
                ordr = 0
            ax2.set_ylabel(r'$|\overline{n}_{n,m}^{%i}| / 10^{%i}$' % (i, ordr))
            ax2.plot(singlesdat[:, 0], np.abs(tmp) / (10 ** ordr), linewidth=0.5)
            ax2.set_xlabel(r'$J\,t$')
            plt.tight_layout()
            pp.savefig(f)
            f.clf()
            plt.close()
            # do the double log plot
            de = np.abs(singlesinfo[i, infoind] - singlesinfo[i, infoind + 1])
            linar = np.zeros(len(singlesdat[:, 0]), dtype=np.float64)
            linar[0] = 0
            linar[1:] = 2 / (singlesdat[1:, 0] * de)
            plt.xlabel(r'$J\,t$')
            plt.ylabel(r'relative average $|n_{n,m}^{%i}|$' % (i))
            plt.loglog(singlesdat[1:, 0], np.abs(tmp / np.abs(comp[0]))[1:], singlesdat[1:, 0], linar[1:], lw=0.5)
            pp.savefig()
            plt.clf()
            plt.close()
        print('.', end='', flush=True)
    diagdat = np.loadtxt('./data/diagsingles.dat')

    if os.path.isfile('./data/energy.dat') and os.path.isfile('./data/hamiltonian_eigvals.dat'):
        ### look for energy - this works because the energies are sorted
        engy = np.loadtxt('./data/energy.dat')
        eigengy = np.loadtxt('./data/hamiltonian_eigvals.dat')
        diff = 0
        for l in range(sysVar.dim):
            if np.abs(eigengy[l, 1] - engy[0, 1]) > diff and l != 0:
                eind = l - 1
                break
            else:
                diff = np.abs(eigengy[l, 1] - engy[0, 1])
        if eind < 15:
            loran = 0
        else:
            loran = eind - 15

    for i in range(sysVar.m):
        if os.path.isfile('./data/energy.dat') and os.path.isfile('./data/hamiltonian_eigvals.dat'):
            plt.title(r'Diagonal weighted elements of $n_{%i}$ in spectral decomp.' % (i))
            lo = np.int32(sysVar.dim * i)
            hi = np.int32(lo + sysVar.dim)
            plt.ylabel(r'$|n%i_{E}|$' % (i))
            plt.xlabel(r'$E / J$')
            # plt.plot(diagdat[lo:hi,1], diagdat[lo:hi,2],linestyle='none',marker='o',ms=0.5)
            plt.plot(diagdat[lo + loran:hi, 1][:30], diagdat[lo + loran:hi, 2][:30], marker='o', ms=2)
            plt.axvline(x=engy[0, 1], linewidth=0.8, color='red')

            ###inlay
            a = plt.axes([0.18, 0.6, 0.28, 0.28])
            a.plot(diagdat[lo:hi - 300, 1], diagdat[lo:hi - 300, 2], marker='o', ms=0.6, ls='none')
            a.set_xticks([])
            a.set_yticks([])

            pp.savefig()
            plt.clf()
            if os.path.isfile('./data/occ' + str(i) + '_re.dat'):
                occmat = np.loadtxt('./data/occ' + str(i) + '_re.dat')
                diags = np.zeros(sysVar.dim)

                ### large plot
                plt.title(r'Diagonal elements of $n_{%i}$ in spectral decomposition' % (i))
                plt.ylabel(r'$|n%i_{E}|$' % (i))
                plt.xlabel(r'$E / J$')
                for el in range(sysVar.dim):
                    diags[el] = occmat[el, el]

                plt.plot(diagdat[lo + loran:hi, 1][:30], diags[loran:][:30], marker='o', ms=2)
                plt.axvline(x=engy[0, 1], linewidth=0.8, color='red')
                ### inlay
                a = plt.axes([0.18, 0.6, 0.28, 0.28])
                a.plot(diagdat[lo:hi - 50, 1], diags[:-50], marker='o', ms=0.5, ls='none')
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
            # plt.plot(diagdat[lo:hi,1], diagdat[lo:hi,2],linestyle='none',marker='o',ms=0.5)
            plt.plot(diagdat[lo:hi - 200, 1], diagdat[lo:hi - 200, 2], marker='o', ms=0.6, ls='none')
            plt.tight_layout()
            pp.savefig()
            plt.clf()

    print('.', end='', flush=True)
    pp.close()
    plt.close()
    print(' done!')


def plotTimescale(sysVar):
    print("Plotting difference to mean.", end='', flush=True)
    pp = PdfPages(sysVar.dataFolder + 'lndiff.pdf')
    params = {
        'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
    }
    plt.rcParams['agg.path.chunksize'] = 0
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

    ### get the characteristic energy difference of the system
    if sysVar.boolEngyStore:
        engys = np.loadtxt('./data/hamiltonian_eigvals.dat')
        enscale = np.abs(engys[0, 1] - engys[-1, 1]) / sysVar.dim
        del engys

    loavgpercent = sysVar.plotLoAvgPerc  # percentage of time evolution to start averaging
    loavgind = int(loavgpercent * sysVar.dataPoints)  # index to start at when calculating average and stddev
    loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale), 2)

    occfile = './data/occupation.dat'
    occ_array = np.loadtxt(occfile)
    # multiply step array with time scale
    step_array = occ_array[:, 0] * sysVar.plotTimeScale

    entfile = './data/entropy.dat'
    ent_array = np.loadtxt(entfile)

    occavg = []
    for i in range(sysVar.m):
        occavg.append(np.mean(occ_array[loavgind:, i + 1], dtype=np.float64))

    entavg = np.mean(ent_array[loavgind:, 1], dtype=np.float64)

    odiff = []
    for i in range(sysVar.m):
        odiff.append(occ_array[:, i + 1] - occavg[i])

    entdiff = ent_array[:, 1] - entavg

    for i in range(sysVar.m):
        plt.ylabel(r'$\Delta n_%i$' % (i))
        plt.xlabel(r'$J\,t$')
        plt.plot(occ_array[:, 0], odiff[i], lw=0.5)
        if sysVar.boolEngyStore:
            plt.axvline(enscale, color='red', lw=0.5)
        pp.savefig()
        plt.clf()
        plt.ylabel(r'$| \Delta n_%i |$' % (i))
        plt.xlabel(r'$J\,t$')
        plt.ylim(ymin=1e-3)
        plt.semilogy(occ_array[:, 0], np.abs(odiff[i]), lw=0.5)
        if sysVar.boolEngyStore:
            plt.axvline(enscale, color='red', lw=0.5)
        plt.tight_layout()
        pp.savefig()
        plt.clf()
        print('.', end='', flush=True)

    plt.ylabel(r'$\Delta S_{ss}$')
    plt.xlabel(r'$J\,t$')
    plt.plot(occ_array[:, 0], entdiff[:], lw=0.5)
    if sysVar.boolEngyStore:
        plt.axvline(enscale, color='red', lw=0.5)
    pp.savefig()
    plt.clf()
    plt.ylabel(r'$| \Delta S_{ss} |$')
    plt.xlabel(r'$J\,t$')
    plt.ylim(ymin=1e-3)
    plt.semilogy(occ_array[:, 0], np.abs(entdiff[:]), lw=0.5)
    if sysVar.boolEngyStore:
        plt.axvline(enscale, color='red', lw=0.5)
    plt.tight_layout()
    pp.savefig()
    plt.clf()
    print('.', end='', flush=True)
    pp.close()
    print(" done!")


def plotMatrix(fPath):
    fName = fPath.split('/')[-1:][0]
    pp = PdfPages(sysVar.dataFolder + fName[:-4] + '.pdf')
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    plt.title('absolute value of matrix')
    dabs = np.loadtxt(fPath)
    plt.xlabel('column')
    plt.ylabel('row')
    cmapVar = plt.cm.Reds
    cmapVar.set_under(color='black')
    plt.imshow(dabs, cmap=cmapVar, interpolation='none', vmin=1e-12)
    pp.savefig()
    pp.close()
    plt.close()