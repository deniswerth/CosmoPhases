import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import *
import time
from numpy import linalg as LA
from scipy import integrate
import timeout_decorator
from scipy.integrate import quad

from phase_transition import Phase_Transition
from potential import Finite_Temperature_Potential


class bubble_wall_velocity():
    """
    This class compute the bubble wall velocity for 
    a given point (mDelta, kappa, lambdaDelta) in 
    parameter space. It computes the entire phase
    transition dynamics for a given point.
    """
    
    def __init__(self, mDelta, kappa, LambdaDelta):
        #Free parameters
        self.mDelta = mDelta
        self.kappa = kappa
        self.LambdaDelta = LambdaDelta
        
        #SM parameters
        self.thetaW = np.arcsin(np.sqrt(0.23129)) #Weinberge weak angle
        self.e = np.sqrt(4*np.pi*7.2973525664e-3) #Electric charge in natural units
        self.g = self.e/np.sin(self.thetaW) #Weak coupling constants
        self.gprime = self.e/np.cos(self.thetaW)
        self.yt = 1 #Top quark Yukawa coupling constant
        self.gs = 1.23 #Strong coupling constant
        self.mH0 = 125.123 #Higgs boson mass
        self.v0 = 246.22 #Higgs boson vev
        self.muHSq = self.mH0**2/2 
        self.LambdaH = self.mH0**2/(2*self.v0**2)
        self.muDeltaSq = self.kappa*self.v0**2/2 - self.mDelta**2 
        self.c2b = 1/6
        
        #Intergration parameters
        self.n = 1000

        #Gauge boson contribution
        self.overdamped_evolution = False
        self.at_fluid_equation_level = False


    def PT_dynamic(self):
        """
        This function determines the phase transition
        dynamics ie the nucleation temperature Tn, the energy gap
        between the two phases DeltaV and the bubble profile phi0
        and Lw.
        """
        pt = Phase_Transition(self.mDelta, self.LambdaDelta, self.kappa)
        Tn, DeltaV, h0, sigma0, v_T = pt.phase_transition_nucleation()
        z, phi, dphi, phi_fit, phi0, Lw = pt.bubble_profile()
        return Tn, DeltaV, phi0, Lw


    def phi_dphi(self, phi0, Lw):
        """
        This function returns the wall profile and
        its derivative as well as the z range for 
        integration.
        """
        z0 = -10*Lw
        zf = 10*Lw
        n = self.n
        z_eval = np.linspace(z0, zf, n)
        phi = phi0/2*(np.tanh(z_eval/Lw) + 1)
        dphi = phi0/(2*Lw)*1/(np.cosh(z_eval/Lw))**2
        return phi, dphi, z_eval


    def fluid_equation_solutions(self):
        """
        This function returns the solutions to the
        fluid equations.
        """
        Tn, DeltaV, phi0, Lw = self.PT_dynamic()
        phi, dphi, z_eval = self.phi_dphi(phi0, Lw)
        
        #C1 coefficients for bosons
        Pi_H = Tn**2*( 3*self.g**2/16 + self.gprime**2/16 + self.yt**2/4 + self.LambdaH/2 + self.kappa/24 )
        mH_Sq = np.abs(-self.muHSq + 3*self.LambdaH*phi**2 + Pi_H)
        c1H = np.abs(np.log(Tn/np.sqrt(np.abs(mH_Sq)))/(2*np.pi**2))
        Pi_Delta = Tn**2*( self.kappa/6 + self.LambdaDelta/2 )
        mDelta_Sq = np.abs(-self.muDeltaSq + 3*self.LambdaDelta*phi**2 + Pi_Delta)
        c1Delta = np.log(Tn/np.sqrt(np.abs(mDelta_Sq)))/(2*np.pi**2)

        #Collision matrix
        Gamma = Tn*np.asarray([[5.8e-4*self.gs**4 + 3.21e-4*self.yt**2*self.gs**2 + 9.6e-5*self.yt**4, -9.81e-4*self.yt**2*self.gs**2 - 4.5e-4*self.yt**4, 0], 
                              [ -1.17e-4*self.yt**2*self.gs**2 - 5.4e-5*self.yt**2, 9.81e-4*self.yt**2*self.gs**2 + 4.5e-4*self.yt**4, 5.0e-5*self.kappa**2],
                              [0., 5.0e-5*self.kappa**2, 0.]])
        
        #Source term
        M = np.asarray([[self.yt**2],
                        [6*self.LambdaH],
                        [self.kappa]])/(2*Tn)
        c1f = np.log(2)/(2*np.pi**2)*np.ones(np.shape(phi))
        C1 = np.asarray([[c1f], 
                         [c1H],
                         [c1Delta]])
        Source = C1*M*phi*dphi

        #deltas
        delta = np.matmul(LA.inv(Gamma), Source)
        delta = np.asarray([delta[0, 0], delta[1, 1], delta[2, 2]])
        
        #For integrand
        M = np.asarray([self.yt**2*np.ones(np.shape(phi)), 6*self.LambdaH*np.ones(np.shape(phi)), 6*self.LambdaDelta*np.ones(np.shape(phi))])
        C1 = np.asarray([c1f, c1H, c1Delta])
        
        if self.at_fluid_equation_level:
            mW_Sq = (self.g**2 + self.gprime**2/2)/4*phi**2 #Average of mW and mZ
            c1W = np.log(Tn/np.sqrt(np.abs(mW_Sq)))/(2*np.pi**2)

            Gamma = Tn*np.asarray([[5.8e-4*self.gs**4 + 3.21e-4*self.yt**2*self.gs**2 + 9.6e-5*self.yt**4, -9.81e-4*self.yt**2*self.gs**2 - 4.5e-4*self.yt**4, 0., 4.6e-4*self.g**4], 
                              [-1.17e-4*self.yt**2*self.gs**2 - 5.4e-5*self.yt**2, 9.81e-4*self.yt**2*self.gs**2 + 4.5e-4*self.yt**4, 5.0e-5*self.kappa**2, 0.],
                              [0., 5.0e-5*self.kappa**2, 0., 0.],
                              [6.5e-3*self.g**4, 0., 0., 1.3e-2*self.g**4 + 9.6e-4*self.gs**2*self.g**2]])
            M = np.asarray([[self.yt**2],
                            [6*self.LambdaH],
                            [self.kappa],
                            [(self.g**2 + self.gprime**2/2)/2]])/(2*Tn)
            c1f = np.log(2)/(2*np.pi**2)*np.ones(np.shape(phi))
            C1 = np.asarray([[c1f], 
                            [c1H],
                            [c1Delta],
                            [c1W]])
            Source = C1*M*phi*dphi
            delta = np.matmul(LA.inv(Gamma), Source)
            delta = np.asarray([delta[0, 0], delta[1, 1], delta[2, 2], delta[3, 3]])
            M = np.asarray([self.yt**2*np.ones(np.shape(phi)), 6*self.LambdaH*np.ones(np.shape(phi)), 6*self.LambdaDelta*np.ones(np.shape(phi)), (self.g**2 + self.gprime**2/2)/2*np.ones(np.shape(phi))])
            C1 = np.asarray([c1f, c1H, c1Delta, c1W])

        return delta, M, phi, dphi, z_eval, C1, Tn, DeltaV, phi0, Lw


    @timeout_decorator.timeout(10, timeout_exception = StopIteration)
    def velocity(self):
        """
        This function returns the bubble wall velocity for
        a given point in parameter space. It uses all the
        other functions.
        """
        delta, M, phi, dphi, z_eval, C1, Tn, DeltaV, phi0, Lw = self.fluid_equation_solutions()
        dof = np.asarray([12*np.ones(np.shape(phi)), 1*np.ones(np.shape(phi)), 1*np.ones(np.shape(phi))]) #degrees of freedom of each particle species
        if self.at_fluid_equation_level:
            dof = np.asarray([12*np.ones(np.shape(phi)), 1*np.ones(np.shape(phi)), 1*np.ones(np.shape(phi)), 9*np.ones(np.shape(phi))])
        integrand = Tn/2*M*phi*dphi*dof*delta*C1
        integration = np.trapz(integrand, x = z_eval)
        integral = np.sum(integration)

        if self.overdamped_evolution:
            N = 9
            mD2 = 11/6*self.g**2*Tn**2
            alpha = (self.g**2 + self.gprime**2/2)/4
            integrandW = 4*alpha**2*phi**2*dphi**2/(alpha*phi**2 + mD2)**2
            integral += N*mD2*Tn/(256*np.pi)*np.trapz(integrandW, x = z_eval)

        C = -DeltaV/integral
        vw = np.sqrt(C**2/(1+C**2))
        return vw, phi0, Tn, Lw, DeltaV


    def distribution_functions(self, T):
        """
        This function returns the distribution functions of the heavy
        particles as well as the deviations from the equilibrium.
        """
        
        delta, M, phi, dphi, z_eval, C1, Tn, DeltaV, phi0, Lw = self.fluid_equation_solutions()
        
        Pi_h = T**2*(self.gprime**2/16 + 3*self.g**2/16 + self.LambdaH/2 + self.yt*2/4 + self.kappa/24)
        m_h2 = 3*self.LambdaH*phi**2 + Pi_h
        
        Pi_Delta = T**2*(self.gprime**2/16 + 3*self.g**2/16 + self.kappa/3 + self.LambdaDelta/4)
        m_Delta2 = 3*self.LambdaDelta*phi**2 + Pi_Delta
        
        Pi_t = self.gs**2*T**2/6
        m_t2 = self.yt**2/2*phi**2 + Pi_t
        
        
        Sq_mass = np.asarray([[m_t2],
                              [m_h2],
                              [m_Delta2]])
        
        Sq_mass = np.reshape(Sq_mass, newshape = (3, self.n))
        
        distribution_functions = np.asarray([[(np.exp(np.sqrt(T + Sq_mass[0])/T) + 1)**(-1)], 
                        [(np.exp(np.sqrt(T + Sq_mass[1])/T) - 1)**(-1)],
                        [(np.exp(np.sqrt(T + Sq_mass[2])/T) - 1)**(-1)]])
        
        dm_dz = np.asarray([[self.yt**2*phi*dphi],
                            [6*self.LambdaH*phi*dphi],
                            [6*self.LambdaDelta*phi*dphi]])
        
        dm_dz = np.reshape(dm_dz, newshape = (3, self.n))
    
        dE_dz = np.asarray([[np.sqrt(Sq_mass[0])/np.sqrt(T**2 + Sq_mass[0])*dm_dz[0]],
                            [np.sqrt(Sq_mass[1])/np.sqrt(T**2 + Sq_mass[1])*dm_dz[1]],
                            [np.sqrt(Sq_mass[2])/np.sqrt(T**2 + Sq_mass[2])*dm_dz[2]]])
        
        dE_dz = np.reshape(dE_dz, newshape = (3, self.n))
        
        delta_distribution_functions = np.asarray([[dE_dz[0]/Pi_t/T*np.exp(np.sqrt(T + Sq_mass[0])/T)/(np.exp(np.sqrt(T + Sq_mass[0])/T) + 1)**2],
                               [dE_dz[1]/Pi_h/T*np.exp(np.sqrt(T + Sq_mass[1])/T)/(np.exp(np.sqrt(T + Sq_mass[1])/T) - 1)**2],
                               [dE_dz[2]/Pi_Delta/T*np.exp(np.sqrt(T + Sq_mass[2])/T)/(np.exp(np.sqrt(T + Sq_mass[2])/T) - 1)**2]])*delta/T
        
        fs = np.asarray([distribution_functions[0, 0], distribution_functions[1, 0], distribution_functions[2, 0]])
        delta_fs = np.asarray([delta_distribution_functions[0, 0], delta_distribution_functions[1, 0], delta_distribution_functions[2, 0]])
        return fs, delta_fs, dE_dz, z_eval


    def plot_velocity_sensitivity(self):
        """
        This function generates three plots showing the 
        sensitivity of the velocity to the 
        (Tn, phi0, Lw) parameters for a given point in 
        the parameter space.
        """
        mDelta, kappa, LambdaDelta = self.mDelta, self.kappa, self.LambdaDelta
        Tn, DeltaV, phi0, Lw = self.PT_dynamic()
        Tn, phi0, Lw = float(Tn), float(phi0), float(Lw)
    
        Tn1 = np.linspace(Tn - 20, Tn + 20, 50)
        phi01 = np.linspace(phi0 - 20, phi0 + 20, 50)
        Lw1 = Lw
        bwvv = bubble_wall_velocity_vectorized(Tn = Tn1, phi0 = phi01, Lw = Lw1, mDelta = mDelta, kappa = kappa, LambdaDelta = LambdaDelta)
        vw1 = bwvv.colormap()
    
        Tn2 = Tn
        phi02 = np.linspace(phi0 - 20, phi0 + 20, 50)
        Lw2 = np.linspace(Lw - 0.05, Lw + 0.05, 50)
        bwvv = bubble_wall_velocity_vectorized(Tn = Tn2, phi0 = phi02, Lw = Lw2, mDelta = mDelta, kappa = kappa, LambdaDelta = LambdaDelta)
        vw2 = bwvv.colormap()
    
        Tn3 = np.linspace(Tn - 20, Tn + 20, 50)
        phi03 = phi0
        Lw3 = np.linspace(Lw - 0.05, Lw + 0.05, 50)
        bwvv = bubble_wall_velocity_vectorized(Tn = Tn3, phi0 = phi03, Lw = Lw3, mDelta = mDelta, kappa = kappa, LambdaDelta = LambdaDelta)
        vw3 = bwvv.colormap()
    
        vw = np.array([vw1, vw2, vw3])
        v1 = np.reshape(vw1, (1,np.product(vw1.shape)))[0]
        v2 = np.reshape(vw2, (1,np.product(vw2.shape)))[0]
        v3 = np.reshape(vw3, (1,np.product(vw3.shape)))[0]
        v = np.concatenate([np.concatenate([v1, v2]), v3])
    
        fig, axs = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

        ax1, ax2, ax3 = axs[0], axs[1], axs[2]

        ax1.contourf(Tn1, phi01, vw[0], 10, vmin = np.min(v), vmax = np.max(v))
        ax1.set_xlabel("$T_n$ [GeV]", fontsize = 15)
        ax1.set_ylabel("$\phi_0$ [GeV]", fontsize = 15)
        ax1.plot(Tn, phi0, ".", markersize = 10, markeredgecolor = "w", color = "C1")
        ax1.set_xlim(Tn1[0], Tn1[-1])
        ax1.set_ylim(phi01[0], phi01[-1])

        ax2.contourf(Lw2, phi02, vw[1], 10, vmin = np.min(v), vmax = np.max(v))
        ax2.set_xlabel("$L_w$", fontsize = 15)
        ax2.set_ylabel("$\phi_0$ [GeV]", fontsize = 15)
        ax2.plot(Lw, phi0, ".", markersize = 10, markeredgecolor = "w", color = "C1")

        ax3.contourf(Tn3, Lw3, vw[2], 10, vmin = np.min(v), vmax = np.max(v))
        ax3.set_xlabel("$T_n$ [GeV]", fontsize = 15)
        ax3.set_ylabel("$L_w$", fontsize = 15)
        ax3.plot(Tn, Lw, ".", markersize = 10, markeredgecolor = "w", color = "C1")

        im = ax1.scatter(np.zeros(np.shape(v)), np.zeros(np.shape(v)), c = v)
        cb = fig.colorbar(im, ax = axs, shrink = 0.8, location = "top")
        cb.set_label(label='$v_w$', fontsize = 15)
        cb.ax.xaxis.set_ticks_position('bottom')
        plt.show()
    
    
#----------------------------------------------------
#----------------------------------------------------
#----------------------------------------------------



class bubble_wall_velocity_vectorized():
    """
    This class compute the bubble wall velocity for 
    various values of (Tn, phi0, Lw). The code
    is written in a way such that those values can be 
    vectorized. It choose the accurate 
    value of DeltaV according to the chosen point
    (mDelta, kappa, lambdaDelta) in the parameter space.
    """
    
    def __init__(self, Tn, phi0, Lw, mDelta, kappa, LambdaDelta):
        #Free parameters
        self.Tn = Tn
        self.phi0 = phi0
        self.Lw = Lw
        self.mDelta = mDelta
        self.kappa = kappa
        self.LambdaDelta = LambdaDelta
        
        #SM parameters
        self.thetaW = np.arcsin(np.sqrt(0.23129)) #Weinberge weak angle
        self.e = np.sqrt(4*np.pi*7.2973525664e-3) #Electric charge in natural units
        self.g = self.e/np.sin(self.thetaW) #Weak coupling constants
        self.gprime = self.e/np.cos(self.thetaW)
        self.yt = 1 #Top quark Yukawa coupling constant
        self.gs = 1.23 #Strong coupling constant
        self.mH0 = 125.123 #Higgs boson mass
        self.v0 = 246.22 #Higgs boson vev
        self.muHSq = self.mH0**2/2 
        self.LambdaH = self.mH0**2/(2*self.v0**2)
        self.muDeltaSq = self.kappa*self.v0**2/2 - self.mDelta**2 
        self.c2b = 1/6
        
        #Phase transition parameter
        self.DeltaV = self.PT_dynamic()
        
        
    def one_step_first_order(self):
        """
        This function returns true if the phase
        transition is one-step and first order, and false otherwise.
        """
        ftp = Finite_Temperature_Potential(self.mDelta, self.LambdaDelta, self.kappa)
        PT = ftp.calcTcTrans()
        if PT == None or PT == []:
            return False
        if len(PT) == 1:
            if PT[0]['trantype'] == 1:
                return True
        else:
            return False
        
        
    def PT_dynamic(self):
        """
        This function determines the phase transition
        dynamics the nucleation temperature Tn, the energy gap
        between the two phases DeltaV and the bubble profile phi0
        and Lw.
        """
        PT = Phase_Transition(self.mDelta, self.LambdaDelta, self.kappa)
        Tn, DeltaV, h0, sigma0, v_T = PT.phase_transition_nucleation()
        z, phi, dphi, phi_fit, phi0, Lw = PT.bubble_profile()
        print("Tn = " + str(Tn))
        print("phi0 = " + str(phi0))
        print("Lw = " + str(Lw))
        print("The phase transition strength is " + str(phi0/Tn))
        return DeltaV
    
    
    def phi_dphi(self, phi0, Lw):
        """
        This function returns the wall profile and
        its derivative as well as the z range for 
        integration.
        """
        z0 = -10*Lw
        zf = 10*Lw
        n = 1000
        z_eval = np.linspace(z0, zf, n)
        phi = phi0/2*(np.tanh(z_eval/Lw) + 1)
        dphi = phi0/(2*Lw)*1/(np.cosh(z_eval/Lw))**2
        return phi, dphi, z_eval
    
    
    def fluid_equation_solutions(self, Tn, phi0, Lw):
        """
        This function returns the solutions to the
        fluid equations.
        """
        phi, dphi, z_eval = self.phi_dphi(phi0, Lw)
        
        #C1 coefficients for bosons
        Pi_H = Tn**2*( 3*self.g**2/16 + self.gprime**2/16 + self.yt**2/4 + self.LambdaH/2 + self.kappa/24 )
        mH_Sq = np.abs(-self.muHSq + 3*self.LambdaH*phi**2 + Pi_H)
        c1H = np.abs(np.log(Tn/np.sqrt(np.abs(mH_Sq)))/(2*np.pi**2))
        Pi_Delta = Tn**2*( self.kappa/6 + self.LambdaDelta/2 )
        mDelta_Sq = np.abs(-self.muDeltaSq + 3*self.LambdaDelta*phi**2 + Pi_Delta)
        c1Delta = np.log(Tn/np.sqrt(np.abs(mDelta_Sq)))/(2*np.pi**2)

        #Collision matrix
        Gamma = Tn*np.asarray([[5.8e-4*self.gs**4 + 3.21e-4*self.yt**2*self.gs**2 + 9.6e-5*self.yt**4, -9.81e-4*self.yt**2*self.gs**2 - 4.5e-4*self.yt**4, 0], 
                              [-1.17e-4*self.yt**2*self.gs**2 - 5.4e-5*self.yt**2, 9.81e-4*self.yt**2*self.gs**2 + 4.5e-4*self.yt**4, 5.0e-5*self.kappa**2],
                              [0., 5.0e-5*self.kappa**2, 0.]])
        
        #Source term
        M = np.asarray([[self.yt**2],
                        [6*self.LambdaH],
                        [self.kappa]])/(2*Tn)
        c1f = np.log(2)/(2*np.pi**2)*np.ones(np.shape(phi))
        C1 = np.asarray([[c1f], 
                         [c1H],
                         [c1Delta]])
        Source = C1*M*phi*dphi

        #deltas
        delta = np.matmul(LA.inv(Gamma), Source)
        delta = np.asarray([delta[0, 0], delta[1, 1], delta[2, 2]])
        
        #For integrand
        M = np.asarray([self.yt**2*np.ones(np.shape(phi)), 6*self.LambdaH*np.ones(np.shape(phi)), 6*self.LambdaDelta*np.ones(np.shape(phi))])
        C1 = np.asarray([c1f, c1H, c1Delta])
        
        return delta, M, phi, dphi, z_eval, C1
    
    
    def velocity(self, Tn, phi0, Lw):
        """
        This function returns the bubble wall velocity for
        a given point in parameter space. It uses all the
        other functions.
        """
        delta, M, phi, dphi, z_eval, C1 = self.fluid_equation_solutions(Tn, phi0, Lw)
        dof = np.asarray([12*np.ones(np.shape(phi)), 1*np.ones(np.shape(phi)), 1*np.ones(np.shape(phi))]) #degrees of freedom of each particle species
        integrand = Tn/2*M*phi*dphi*dof*delta*C1
        integration = np.trapz(integrand, x = z_eval)
        integral = np.sum(integration)
        C = -self.DeltaV/integral
        vw = np.sqrt(C**2/(1+C**2))
        return vw
    
    
    def colormap(self):
        """
        This function returns a 2d array of the velocity
        according to the given Tn, phi0 and Lw.
        """
        if type(self.Tn) == float:
            n_phi0 = len(self.phi0)
            n_Lw = len(self.Lw)
            vw = np.zeros((n_phi0, n_Lw))
            for i in range(n_phi0):
                for j in range(n_Lw):
                    vw[i, j] = self.velocity(self.Tn, self.phi0[i], self.Lw[j])
                    
        if type(self.phi0) == float:
            n_Tn = len(self.Tn)
            n_Lw = len(self.Lw)
            vw = np.zeros((n_Tn, n_Lw))
            for i in range(n_Tn):
                for j in range(n_Lw):
                    vw[i, j] = self.velocity(self.Tn[i], self.phi0, self.Lw[j])
        
        if type(self.Lw) == float:
            n_Tn = len(self.Tn)
            n_phi0 = len(self.phi0)
            vw = np.zeros((n_Tn, n_phi0))
            for i in range(n_Tn):
                for j in range(n_phi0):
                    vw[i, j] = self.velocity(self.Tn[i], self.phi0[j], self.Lw)       
        return vw


