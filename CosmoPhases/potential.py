import numpy as np
from cosmoTransitions import generic_potential


class Finite_Temperature_Potential(generic_potential.generic_potential):
    """
    Finite temperature effective potential including the tree level potential,
    the one-loop Coleman-Weinberg potential, and the one-loop thermal potential.
    """

    def init(self, mDelta, LambdaDelta, kappa):
        """
        kappa       : portal coupling constant (free parameter)
        mDelta      : Higgs triplet mass (free parameter)
        LambdaDelta : Triplet self-coupling (free parameter)
        v0          : Higgs vev (fixed value)
        mH0         : Measured Higgs mass (fixed value)
        """
        
        #Number of dimensions
        self.Ndim = 2

        #Class parameters
        self.v0 = 246.22 #GeV
        self.mH0 = 125.123 #GeV
        self.kappa = kappa
        self.mDelta = mDelta
        self.LambdaDelta = LambdaDelta
        
        #Fixed tree-level potential coupling constants
        self.LambdaH = self.mH0**2/(2*self.v0**2)
        self.muH2 = self.mH0**2/2
        self.muDelta2 = self.kappa*self.v0**2/2 - mDelta**2
        
        #SM coupling constants in natural units
        self.thetaW = np.arcsin(np.sqrt(0.23129))#Weinberg angle
        self.e = np.sqrt(4*np.pi*7.2973525664e-3) #electric charge
        self.g = self.e/np.sin(self.thetaW) #eletroweak coupling constant
        self.gprime = self.e/np.cos(self.thetaW) #eletroweak coupling constant
        self.yt = 1 #top quark Yukawa coupling constant
        self.gs = 1.23 #strong coupling constant

        #Print wrong vacuum if (v0, 0) is not the global minimum at zero temperature
        if self.v0*self.mH0/np.sqrt(2) > 1/np.sqrt(self.LambdaDelta)*( self.kappa*self.v0**2/2 - self.mDelta**2 ):
            print("The point (h, sigma) = (v0, 0) is the global minimum.") 
        else:
            print("The point (h, sigma) = (v0, 0) is not the global minimum.") 

        #Print second minimum appearing along the sigma direction
        if self.mH0**2*self.LambdaDelta < self.kappa*(self.kappa*self.v0**2/2 - self.mDelta**2):
            print("Second minimum appearing along the sigma direction") 



    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return (np.array([X])[...,0] < -5.0).any()


    def V0(self, X):
        """
        Returns the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        X = np.asanyarray(X)
            
        h, sigma = X[...,0], X[...,1]
        
        muH2 = self.muH2
        LambdaH = self.LambdaH
        muDelta2 = self.muDelta2
        LambdaDelta = self.LambdaDelta
        kappa = self.kappa
        
        r = - muH2*h*h/2 + LambdaH*h*h*h*h/4
        r += - muDelta2*sigma*sigma/2 + LambdaDelta*sigma*sigma*sigma*sigma/4
        r += kappa*h*h*sigma*sigma/4
        return r


    def boson_massSq(self, X, T):
        """
        Returns the square masses of the bosons in the so-called on-shell
        renormalization. Thermal corrections to the masses are taken into account.
        Absolute values were added for masses that can be negative for certain values
        of the fields h and sigma.
        """
        X = np.array(X)
        h, sigma = X[...,0], X[...,1]
        
        v0 = self.v0
        g = self.g
        gprime = self.gprime
        yt = self.yt
        gs = self.gs
        
        muH2 = self.muH2
        muDelta2 = self.muDelta2
        LambdaH = self.LambdaH
        LambdaDelta = self.LambdaDelta
        kappa = self.kappa
        
        #Thermal corrections
        Pi_H = T**2*( 3*g**2/16 + gprime**2/16 + yt**2/4 + 3*LambdaH/24 + kappa/24 ) #SM-like Higgs boson 
        Pi_Delta = T**2*( kappa/16 + LambdaDelta/12 ) #Higgs triplet
        Pi_W = 11*g**2*T**2/6 #W boson
        Pi_Z = 11*g**2*T**2/6 #Z boson
        Pi_gamma = 11*gprime**2*T**2/6 #photon
        Pi_chi = Pi_H #SM-like Goldstone boson 
        Pi_xi = Pi_Delta #Higgs triplet Goldstone boson
        Pi_t = T**2*(gs**2/6 + 3*g**2/64 + gprime**2/64 + yt**2/16) #top quark

        #SM-like Higgs boson and Higgs triplet
        A = -muH2 + 3*LambdaH*h*h + kappa*sigma*sigma/2 + Pi_H
        B = kappa*h*sigma
        C = -muDelta2 + 3*LambdaDelta*sigma*sigma + kappa*h*h/2 + Pi_Delta
        mH_Sq_field = (A + C - np.sqrt( (A - C)**2 + 4*B**2 ) )/2
        mDelta_Sq_field = (A + C + np.sqrt( (A - C)**2 + 4*B**2 ) )/2

        #Goldstone bosons
        mchi_Sq_field = -muH2 + LambdaH*h*h + kappa*sigma*sigma/2 + Pi_chi
        mxi_Sq_field = -muDelta2 + LambdaDelta*sigma*sigma + kappa*h*h/2 + Pi_xi 

        #W boson
        mW_Sq_field = g**2*np.abs(h*h + 4*sigma*sigma)/4

        #Z boson
        A = g**2*h*h/4 + Pi_Z
        B = -g*gprime*h*h
        C = gprime**2*h*h/4 + Pi_gamma
        mZ_Sq_field = (A + C + np.sqrt((A - C)**2 + 4*B**2 ))/2

        #Squared masses at the VEV for (h, sigma) = (v0, 0)
        mH_Sq_vev = -muH2 + 3*LambdaH*v0**2 + 0*h
        mDelta_Sq_vev = -muDelta2 + kappa*v0**2/2 + 0*h
        mchi_Sq_vev = -muH2 + LambdaH*v0**2 + 0*h + 1e-100
        mxi_Sq_vev = -muDelta2 + kappa*v0**2/2 + 0*h + 1e-100
        mW_Sq_vev = g**2*v0**2/4 + 0*h
        mZ_Sq_vev = (g**2 + gprime**2)*v0**2/4 + 0*h

        #With Goldstone Bosons
        dof = np.array([1, 1, 3, 2, 6, 3])
        c = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        MSq = np.array([mH_Sq_field, mDelta_Sq_field, mchi_Sq_field, mxi_Sq_field, mW_Sq_field, mZ_Sq_field])
        MSq_vev = np.array([mH_Sq_vev, mDelta_Sq_vev, mchi_Sq_vev, mxi_Sq_vev, mW_Sq_vev, mZ_Sq_vev])        
        MSq = np.rollaxis(MSq, 0, len(MSq.shape))
        MSq_vev = np.rollaxis(MSq_vev, 0, len(MSq_vev.shape))
        

        #Without Goldstone Bosons
        """
        dof = np.array([1, 1, 6, 3])
        c = np.array([1.5, 1.5, 1.5, 1.5])
        MSq = np.array([mH_Sq_field, mDelta_Sq_field, mW_Sq_field, mZ_Sq_field])
        MSq_vev = np.array([mH_Sq_vev, mDelta_Sq_vev, mW_Sq_vev, mZ_Sq_vev])        
        MSq = np.rollaxis(MSq, 0, len(MSq.shape))
        MSq_vev = np.rollaxis(MSq_vev, 0, len(MSq_vev.shape))
        """

        return MSq, MSq_vev, dof, c


    def fermion_massSq(self, X):
        """
        Returns the square masses of the fermion.
        Note that cosmoTransitions does not allow to 
        have fermion thermal corrections
        """
        X = np.array(X)
        h, sigma = X[...,0], X[...,1]

        #Top quark
        mt_Sq_field = self.yt**2*h*h/2
        mt_Sq_vev = self.yt**2*self.v0**2/2 + 0*h
        
        MSq = np.array([mt_Sq_field])
        MSq_vev = np.array([mt_Sq_vev])
        MSq = np.rollaxis(MSq, 0, len(MSq.shape))
        MSq_vev = np.rollaxis(MSq_vev, 0, len(MSq_vev.shape))
        dof = np.array([12])
        return MSq, MSq_vev, dof


    def V1(self, bosons, fermions):
        """
        Compute the one-loop Coleman-Weinberg potential 
        in the so-called on-shell renormalization.
        """
        MSq, MSq_vev, dof, c = bosons
        y = np.sum(dof * (MSq*MSq*(np.log(np.abs(MSq/MSq_vev) + 1e-100) - c) + 2*MSq*MSq_vev), axis = -1)

        MSq, MSq_vev, dof = fermions
        c = np.asarray([1.5])
        y -= np.sum(dof * (MSq*MSq*(np.log(np.abs(MSq/MSq_vev) + 1e-100) - c) + 2*MSq*MSq_vev), axis = -1)
        
        return y/(64*np.pi**2)