import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib import cm
from random import *
import numpy as np
import time
import json

from phase_transition import Phase_Transition


class scan():
    """
    This class enables to perform a scan
    of the parameter space in the m_Delta-kappa plane
    for a fixed LambdaDelta.
    """
    
    
    def __init__(self, LambdaDelta, save_paths):
        """
        Lamda_Delta : Higgs triplet quartic coupling (usually 0.01)
        save_paths  : 
        """
        #Gaussian parameters
        self.n = 10000
        self.sx = 1/300
        self.sy = 1/2
        self.theta = -np.pi/180

        #Parallelogram parameters
        self.x_step = 1
        self.y_step = 0.1

        #Defining volume
        self.volume = self.parameter_space_volume()
        self.LambdaDelta = LambdaDelta
        self.v0 = 246.22 #GeV
        self.mH0 = 125.123 #GeV

        #Plot the regions where one or two minima appear at tree-level
        self.plot_regions_tree_level = True

        #Path to saving results 
        self.save_paths = save_paths


    def parameter_space_volume(self):
        """
        Generate a 2d parameter space volume.
        """
        volume = np.asarray([[200, 800], [0, 12]])
        return volume


    def parallelogram(self):
        """
        Generate a 2D parallelogram evenly spaced distribution.
        """
        edge = [[100, 1], [500, 1], [570, 12], [920, 12]]
        X, Y = [], []
    
        x0, y0 = edge[0][0], edge[0][1]
        x1, y1 = edge[2][0], edge[2][1] 
        a = (y1 - y0)/(x1 - x0)
        b = y0 - a*x0
             
        for y in np.arange(y0, y1, self.y_step):
            for x in np.arange((y - b)/a, (y - b)/a + edge[1][0] - edge[0][0], self.x_step):
                X.append(x)
                Y.append(y)
        return X, Y


    def gaussian(self):
        """
        Generates a 2D gaussian distribution.
        """
        Ndim = 2
        n = self.n
        volume = self.volume
        xlow, xhigh, ylow, yhigh = volume[0, 0], volume[0, 1], volume[1, 0], volume[1, 1]
        xmean, ymean = (xlow + xhigh)/2, (ylow + yhigh)/2
        sx, sy, n, theta = self.sx, self.sy, self.n, self.theta
        a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
        b = np.sin(2*theta)/(4*sy**2) - np.sin(2*theta)/(4*sx**2)
        c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
        mean = [xmean, ymean]
        cov = [[a, b], [b, c]]
        np.random.seed(1)
        x, y = np.random.multivariate_normal(mean, cov, n).T
        return x, y


    def plot_distribution_2d(self, X, Y):
        """
        Plot the 2d distribution before running the scan.
        """
        Ndim = 2
        volume = self.volume
        plt.plot(X, Y, '.', markersize = 1)
        plt.xlim([volume[0, 0], volume[0, 1]])
        plt.ylim([volume[1, 0], volume[1, 1]])
        plt.xlabel("$m_{\Delta}$ [GeV]", fontsize = 15)
        plt.ylabel("$\kappa$", fontsize = 15)
        
        if self.plot_regions_tree_level:
            X1 = np.linspace(200, 800, 100)
            Y1 = 2*X1**2/self.v0**2 + 2/self.v0**2*np.sqrt(self.mH0**2*self.v0**2/2*self.LambdaDelta)
            Y2 = np.linspace(1, 12, 100)
            X2 = np.sqrt(Y2*self.v0**2/2 - self.mH0**2*self.LambdaDelta/Y2)
            plt.plot(X1, Y1, color = "k", linewidth = 0.5)
            plt.plot(X2, Y2, color = "k", linewidth = 0.5)

        plt.show()


    def defining_scan_points(self, type):
        """
        Defining the scan points.
        type can be "parallelogram" or "gaussian"
        """ 
        volume = self.volume
        if type == "gaussian":
            x, y = self.gaussian()
        if type == "parallelogram":
            x, y = self.parallelogram()

        xnew, ynew = [], []

        if len(x) != len(y):
            raise Exception('len(X) != len(Y)')

        for i in range(len(x)):
            if volume[0, 0] <= x[i] <= volume[0, 1] and volume[1, 0] <= y[i] <= volume[1, 1] and self.mH0**2*self.LambdaDelta >= y[i]*(y[i]*self.v0**2/2 - x[i]**2):
                xnew.append(x[i])
                ynew.append(y[i])
        return np.asarray(xnew), np.asarray(ynew)


    def run(self, X, Y):
        """
        Run the scan by returning X, Y and the phase transition history.
        """
        if len(X) != len(Y):
            raise Exception('len(X) != len(Y)')
        n = len(X)
        History = []
        mDelta = []
        kappa = []
        for i in range(n):
            print("######## i = {}/{} ######## \n ######## mDelta = {}, kappa = {} ########".format(i, n, X[i], Y[i]))
            PT = Phase_Transition(mDelta = X[i], LambdaDelta = self.LambdaDelta, kappa = Y[i])
            try:
                hist = PT.phase_transition_critical()
                if hist == []:
                    hist = [None]
                History.append(hist)
                mDelta.append(X[i])
                kappa.append(Y[i])
            except Exception:
                pass
        return np.asarray(mDelta), np.asarray(kappa), list(History)


    def save(self, X, Y, History):
        """
        Save the files according to save_paths.
        """
        save_paths = self.save_paths
        np.savetxt(save_paths[0], X, fmt = "%s")
        np.savetxt(save_paths[1], Y, fmt = "%s")

        Z = []
        for i in range(len(History)):
            if History[i] == [None]:
                Z.append([[0, [0, 0], [0, 0], 0, 0]])
            else:
                history = []
                for j in range(len(History[i])):
                    hist = []
                    hist.append(float(History[i][j]["Tcrit"]))
                    hist.append(list(History[i][j]["high_vev"]))
                    hist.append(list(History[i][j]["low_vev"]))
                    hist.append(int(History[i][j]["trantype"]))
                    hist.append(float(History[i][j]["Delta_rho"]))
                    history.append(hist)
                Z.append(history)
        np.savetxt(save_paths[2], np.asarray(Z), fmt = "%s")


    def load(self):
        """
        Load mDelta, kappa and History saved in save_paths.
        """
        save_paths = self.save_paths
        mDelta = np.loadtxt(save_paths[0])
        kappa = np.loadtxt(save_paths[1])
        Z = open(save_paths[2], "r")
        lines = Z.readlines()
        History = []
        for line in lines:
            History.append(json.loads(line))
        return mDelta, kappa, History



























    


