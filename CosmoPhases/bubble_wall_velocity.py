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

from phase_transition import Phase_Transition


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
        mH_Sq = -self.muHSq + 3*self.LambdaH*phi**2 + Pi_H
        c1H = np.log(Tn/np.sqrt(np.abs(mH_Sq)))/(2*np.pi**2)
        Pi_Delta = Tn**2*( self.kappa/6 + self.LambdaDelta/2 )
        mDelta_Sq = -self.muDeltaSq + 3*self.LambdaDelta*phi**2 + Pi_Delta
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
        integrand = Tn/2*M*phi*dphi*dof*delta*C1
        integration = np.trapz(integrand, x = z_eval)
        integral = np.sum(integration)
        C = -DeltaV/integral
        vw = np.sqrt(C**2/(1+C**2))
        return vw, phi0, Tn, Lw, DeltaV





