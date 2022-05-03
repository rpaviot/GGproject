from scipy.special import erfc,erf
import numpy as np
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import interp1d


"""Class of HOD model for ELG and LRG"""

class HOD:

    def __init__(self,Mcut):

        self.ELG_centralv = np.vectorize(self.ELG_central, excluded=['self'])
        self.ELG_sattelitev = np.vectorize(self.ELG_sattelite, excluded=['self'])
        self.LRG_sattelitev = np.vectorize(self.LRG_sattelite, excluded=['self'])
        self.massesC = np.logspace(np.log10(Mcut),np.log10(1e15),1000)


    def set_params(self,tracer,p):
        """ Fixed ELG HOD parameters"""
        if tracer == "ELG":
            self.Ac = p[0]
            self.As = p[1]
            self.mu = p[2]
            self.M0 = p[3]
            self.M1 = p[4]
            self.sigma=p[5]
            self.alpha=p[6]
            self.gamma =p[7]
            #self.sigma = 0.08
            #self.alpha = 0.9
            #self.gamma = -1.4
            #self.M0 = 10**(self.mu - 0.05)
            #self.M1 = 10**(self.mu + 0.35)
            massesS = np.logspace(np.log10(self.M0),np.log10(1e16),10000)
            HOD_central = self.ELG_centralv(self.massesC)
            HOD_sattelite = self.ELG_sattelitev(massesS)
            self.splineC = interp1d(self.massesC,HOD_central,kind="cubic",fill_value = 0,bounds_error=False)
            self.splineS = interp1d(massesS,HOD_sattelite,kind="cubic",fill_value = 0,bounds_error=False)

        elif tracer == "LRG":
            self.Mmin = p[0]
            self.M0 = p[1]
            self.M1 = p[2]
            self.sigma = p[3]
            self.alpha = p[4]
            massesS = np.logspace(np.log10(self.M0),np.log10(1e16),10000)
            HOD_central = self.LRG_central(self.massesC)
            self.splineC = interp1d(self.massesC,HOD_central,kind="cubic",fill_value = (0,1),bounds_error=False)
            HOD_sattelite = self.LRG_sattelitev(massesS)
            self.splineS = interp1d(massesS,HOD_sattelite,kind="cubic",fill_value = 0,bounds_error=False)

    # Avila et al. 2020
    
    def ELG_central(self,M):
        x = np.log10(M)
        if (x < self.mu):
            Ncen = self.Ac/(np.sqrt(2*np.pi)*self.sigma)*np.exp(-((x - self.mu)**2)/(2*self.sigma**2))
        else:
            Ncen= self.Ac/(np.sqrt(2*np.pi)*self.sigma)*(M/(10**self.mu))**(self.gamma)
        return Ncen

    def ELG_sattelite(self,M):
        if (M > self.M0):
            Nsat=self.As*((M - self.M0)/self.M1)**(self.alpha)
        else :
            Nsat=0.0
        return Nsat

    #Zhai et al. 2O17 

    def LRG_central(self,M):
        Ncen = (1./2.)*(1 + erf(np.log(M/self.Mmin)/self.sigma))
        return Ncen


    def LRG_sattelite(self,M):
        Ncen = self.splineC(M)
        Nsat=Ncen*(M/self.M1)**(self.alpha)*np.exp(-self.M0/M)
        return Nsat  

#    def LRG_sattelite(self,M):
#        if (M > self.M0):
#            Ncen = self.splineC(M)
#            Nsat=Ncen*((M - self.M0)/self.M1)**(self.alpha)
#        else :
#            Nsat=0.0
#        return Nsat       
