import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib import cm
import numpy as np
from cosmoTransitions import generic_potential
from scipy.optimize import curve_fit
from potential import Finite_Temperature_Potential


class Phase_Transition():
    """
    This class studies the Higgs triplet phase transition according
    to the effective thermal potential given in potential.py
    """

    def __init__(self, mDelta, LambdaDelta, kappa):
        """
        kappa       : portal coupling constant (free parameter)
        mDelta      : Higgs triplet mass (free parameter)
        LambdaDelta : Triplet self-coupling (free parameter)
        v0          : Higgs vev (fixed value)
        mH0         : Measured Higgs mass (fixed value)
        """

        self.v0 = 246.22 #GeV
        self.mH0= 125.123 #GeV
        self.mDelta = mDelta
        self.LambdaDelta = LambdaDelta
        self.kappa = kappa
        self.Ndim = 2
        self.FTP = Finite_Temperature_Potential(self.mDelta, self.LambdaDelta, self.kappa)
        self.LambdaH = self.mH0**2/(2*self.v0**2)

    
    def plot_2d_potential(self, X, Y, V):
        """
        Plot the 2D potential.
        """
        plt.contourf(X, Y, V*10**(-7), 200, cmap = "coolwarm")
        plt.xlabel("$h$ [GeV]", fontsize = 15)
        plt.ylabel("$\sigma$ [GeV]", fontsize = 15)
        plt.colorbar(label = "$V$ [$\\times 10^{7}$ GeV$^4$]")
        plt.show()
        
        
    def plot_1d_potential(self, X, V):
        """
        Plot a 1D potential.
        """
        plt.plot(X, V*10**(-7))
        plt.xlabel("[GeV]", fontsize = 12)
        plt.ylabel("$V$ ($\\times 10^{7}$ GeV$^4$)", fontsize = 12)
        plt.show()
        
        
    def plot_potential(self):
        """
        Plot the potential for a specific temperature.
        """
        FTP = self.FTP
        print("Choose a temperature, T = ")
        T = float(input())
        print("Choose the number of dimensions for plotting, dimension = ")
        dimension = float(input())
        if dimension < 1 or dimension > 2:
            raise Exception('Error in dimension! The value of dimension is: {}'.format(dimension))
            
        if dimension == 2:
            xlow = -300
            xhigh = 300
            ylow = -300
            yhigh = 300
            n = 200
            X = np.linspace(xlow, xhigh, n)[:,None] * np.ones((1, n))
            Y = np.linspace(ylow, yhigh, n)[None,:] * np.ones((n, 1))
            XY = np.rollaxis(np.array([X,Y]), 0, 3)
            Total_Potential = FTP.DVtot(XY, T)

        elif dimension == 1:
            xlow = 0
            xhigh = 300
            ylow = 0
            yhigh = 300
            n = 200
            X_h = np.linspace(xlow, xhigh, n)[:,None] * np.ones((1, n))
            x_h = np.linspace(xlow, xhigh, n)
            Y_h = np.linspace(ylow, yhigh, n)[None,:] * np.zeros((n, 1))
            XY_h = np.rollaxis(np.array([X_h,Y_h]), 0, 3)
            #Defining the 1D-potential along an axis
            Total_Potential_h = FTP.DVtot(XY_h, T)
            Total_Potential_1D_h = np.zeros(n)
            for i in range(n):
                Total_Potential_1D_h[i] = Total_Potential_h[i][0]
            Y_sigma = np.linspace(ylow, yhigh, n)[:,None] * np.ones((1, n))
            x_sigma = np.linspace(ylow, yhigh, n)
            X_sigma = np.linspace(xlow, xhigh, n)[None,:] * np.zeros((n, 1))
            XY_sigma = np.rollaxis(np.array([X_sigma,Y_sigma]), 0, 3)
            #Defining the 1D-potential along an axis
            Total_Potential_sigma = FTP.DVtot(XY_sigma, T)
            Total_Potential_1D_sigma = np.zeros(n)
            for i in range(n):
                Total_Potential_1D_sigma[i] = Total_Potential_sigma[i][0]
    
        if dimension == 2:
            self.plot_2d_potential(X, Y, Total_Potential)
            
        if dimension == 1:
            fig, axs = plt.subplots(2, figsize=(10,10))
            axs[0].plot(x_h, Total_Potential_1D_h*10**(-7))
            axs[0].set_xlim(xlow, xhigh)
            axs[0].set_xlabel("$h$ [GeV]", fontsize = 12)
            axs[0].set_ylabel("$V$ ($\\times 10^{7}$ GeV$^4$)", fontsize = 12)
            axs[0].axhline(y = 0, linestyle = '--', color = "k")
            axs[1].plot(x_sigma, Total_Potential_1D_sigma*10**(-7))
            axs[1].set_xlim(ylow, yhigh)
            axs[1].set_xlabel("$\sigma$ [GeV]", fontsize = 12)
            axs[1].set_ylabel("$V$ ($\\times 10^{7}$ GeV$^4$)", fontsize = 12)
            axs[1].axhline(y = 0, linestyle = '--', color = "k")
            plt.show()

    
    def phase_transition_critical(self):
        """
        Compute the phase transition for a given point
        in the paraneter space. Returns the phase transition study
        at critical point.
        """
        FTP = self.FTP
        FTP.calcTcTrans()
        return FTP.TcTrans

        
    def plot_potential_Tcrit(self):
        """
        Plot the 2D potential at the critical temperature.
        """
        FTP = self.FTP
        PT = self.phase_transition_critical()
        if len(PT) != 1:
            raise Exception('Error! Not a one step phase transition ==> len(PT) is: {}'.format(len(PT)))
        Tc = PT[0]["Tcrit"]
        xlow = -300
        xhigh = 300
        ylow = -200
        yhigh = 200
        n = 200
        X = np.linspace(xlow, xhigh, n)[:,None] * np.ones((1, n))
        Y = np.linspace(ylow, yhigh, n)[None,:] * np.ones((n, 1))
        XY = np.rollaxis(np.array([X,Y]), 0, 3)
        Total_Potential = FTP.DVtot(XY, Tc)
        self.plot_2d_potential(X, Y, Total_Potential)
        
        
    def bubble_profile(self):
        """
        Compute the bubble wall profile the tanh fit. 
        Returns the cosmoTransition profile, the tanh profile, phi0 and Lw.
        """
        FTP = self.FTP
        FTP.findAllTransitions()
        instanton = FTP.TnTrans[0]['instanton']
        if instanton == None:
            raise Exception('Error! Not a first-order phase transition ==> No instanton.')
        z = instanton.profile1D.R
        phi = instanton.profile1D.Phi
        dphi = instanton.profile1D.dPhi
        #Centering the bubble profile to x = 0
        phi_mid = (phi[-1] + phi[0])/2
        i = 0
        while phi_mid > phi[i]:
            i += 1
        phi_mid = phi[i]
        z = z - z[i]
        phi0 = m.TnTrans[0]['low_vev'][0]
        
        #Ansatz
        def func(z, Lw):
            return phi0/2*(1 + np.tanh(z/Lw))
        
        popt, pcov = curve_fit(func, z, phi)
        phi_fit = func(z, *popt)
        Lw = popt[0]
        return z, phi, dphi, phi_fit, phi0, Lw
        

    def phase_transition_nucleation(self):
        """
        Compute the phase transition for a given point
        in the parameter space. Returns the nucleation
        temperature, the potential gap at Tn, phi0, sigma0 (at Tn) and vn/Tn.
        """
        FTP = self.FTP
        FTP.findAllTransitions()
        Tn = FTP.TnTrans[0]['Tnuc']
        minimum = abs(FTP.findMinimum(T = Tn))
        h0 = minimum[0]
        sigma0 = minimum[1]
        DeltaV = FTP.DVtot(minimum, T = Tn)
        v_T = np.sqrt(np.abs(h0**2 + sigma0**2))/Tn
        return Tn, DeltaV, h0, sigma0, v_T
    
    
    def today_universe(self):
        """
        Returns True if the potential has its global minimum along the h axis 
        at the usual Higgs mass at zero temperature.
        """
        FTP = self.FTP
        T = 0
        if FTP.DVtot(FTP.findMinimum(X = [200, 0], T = T), T = T) < FTP.DVtot(FTP.findMinimum(X = [0, 200], T = T), T = T) and abs(FTP.findMinimum(X = [200, 0], T = 0)[0] - FTP.v0) < 10:
            return 1
        else:
            return 0
        
    
    def bound_from_below(self):
        """
        Returns True if the potential is bound from below.
        """
        kappa = self.kappa
        LambdaH = self.LambdaH
        LambdaDelta = self.LambdaDelta
        if kappa < -2*np.sqrt(LambdaH*LambdaDelta):
            return 0
        else:
            return 1
        
        
    def perturbative_regime(self):
        """
        Returns True if we are in the perturbative regime.
        """
        LambdaH = self.LambdaH
        kappa = self.kappa
        if LambdaH > 4*np.pi or kappa > 4*np.pi:
            return 0
        else:
            return 1
        
        
    def plot_phase_transition(self):
        """
        Plot phase transition.
        """
        FTP = self.FTP
        #if not self.today_universe():
        #    raise Exception('Error! Wrong Universe.')
        FTP.plotPhasesPhi()
        plt.xlim(0, 300)
        plt.xlabel("$T$ [GeV]", fontsize = 12)
        plt.ylabel("$\sqrt{h(T)^2 + \sigma(T)^2}$ [GeV]", fontsize = 12)
        plt.show()