import os
import pandas as pd
import scipy.stats
import healpy as hp
import numpy as np
from astropy.io import fits,ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from scipy.interpolate import CubicSpline as CS
import os.path


L0 = lambda x : 1
L2 =  lambda x : (1./2)*(3*x**2-1)
L4 = lambda x : (35.0*x*x*x*x-30.0*x*x+3.0)/8.0
L6 = lambda x : (231*pow(x,6)-315*pow(x,4)+105*pow(x,2)-5)/16.0
L8 = lambda x : (6435*pow(x,8)-12012*pow(x,6) + 6930*pow(x,4) - 1260*pow(x,2) + 35)/128

from numpy.random import default_rng
rng = default_rng()


class randomcat:
    def __init__(self, ra,dec,z,w):
        self.ra = ra
        self.dec = dec
        self.z = z
        self.w = w
        self.ra_min = np.min(ra)
        self.ra_max = np.max(ra)
        self.dec_min = np.min(dec)
        self.dec_max = np.max(dec)

        self.zmin =round(min(z),2)
        self.zmax = round(max(z),2)
        self.redshift_CDF()

    def redshift_CDF(self):
        bins = np.arange(self.zmin,self.zmax,0.005)
        dn, bins = np.histogram(self.z, bins=bins)
        myPDF = dn/np.sum(dn)
        ##dxc = np.diff(bins)[0];   xc = bins[0:-1] + 0.5*dxc
        myCDF = np.zeros_like(bins)
        myCDF[1:] = np.cumsum(myPDF)
        self.spline_inv = CS(myCDF,bins)
    
    """Work only for simplest rectangular footprint without masks"""
    def create_random_easy(self,number):
        random_numbers = rng.uniform(0,1,size=int(number))
        self.z_rand = self.spline_inv(random_numbers)
        self.ra_rand = np.random.uniform(self.ra_min,self.ra_max,len(self.z_rand))
        self.sindec_rand = np.random.uniform(np.sin(self.dec_min*np.pi/180), np.sin(self.dec_max*np.pi/180),len(self.z_rand))
        self.dec_rand  = np.arcsin(self.sindec_rand)*180/np.pi
        self.w_rand = np.ones(len(self.ra_rand))
        return self.ra_rand,self.dec_rand,self.z_rand,self.w_rand



class survey_geometry:

    def __init__(self,*args,**qwargs):

        self.NSIDE = 256
        self.ra = qwargs['RA']
        self.dec = qwargs['DEC']
        self.w = qwargs['W']
        self.z = qwargs['Z']
        self.catalog_type = qwargs['type']

        if self.catalog_type=="source":
            self.g1 = qwargs['g1']
            self.g2 = qwargs['g2']
        else :
            self.g1 = np.zeros(len(self.ra))
            self.g2 = np.zeros(len(self.ra))

        self.maskdata = self.mask()

    def get_data(self):
        return self.w

    def DeclRaToIndex(self):
        return hp.pixelfunc.ang2pix(self.NSIDE,(-self.dec+90.)*np.pi/180.,self.ra*np.pi/180.)

    def IndexToDeclRa(self,index):
        theta,phi=hp.pixelfunc.pix2ang(self.NSIDE,index)
        return (180./np.pi*phi,-(180./np.pi*theta-90))


    def radec2thphi(self,ra,dec):
        return (-dec+90.)*np.pi/180.,ra*np.pi/180.

    def thphi2radec(self,theta,phi):
        return 180./np.pi*phi,-(180./np.pi*theta-90)

    def mask(self):
        npix = hp.nside2npix(self.NSIDE)
        p = np.zeros(npix)
        d = 0 
        index = self.DeclRaToIndex()
        self.index = index
        footprint = np.unique(index)
        self.footprint = footprint
        p[footprint] =+1
        return p

    def get_mask(self):
        return self.maskdata

    def wmask(self):
        npix = hp.nside2npix(self.NSIDE)
        pixl = np.zeros(npix)
        tot = np.zeros(npix)
        index = self.DeclRaToIndex()
        for i in range(0,len(index)):
            pixl[index[i]] +=  1.*self.w[i]
            tot[index[i]] += 1.
        avg_weight = pixl/tot
        avg_weight[np.isnan(avg_weight)] = 0
        return avg_weight


    @staticmethod
    def match(catalog1,catalog2):
        indices = np.where(catalog2.maskdata[catalog1.index] !=0)
        catalog1.ra = catalog1.ra[indices]
        catalog1.dec = catalog1.dec[indices]
        catalog1.z = catalog1.z[indices]
        catalog1.w = catalog1.w[indices]
        catalog1.g1 = catalog1.g1[indices]
        catalog1.g2 = catalog1.g2[indices]
        catalog1.mask()


        indices = np.where(catalog1.maskdata[catalog2.index] !=0)
        catalog2.ra = catalog2.ra[indices]
        catalog2.dec = catalog2.dec[indices]
        catalog2.z = catalog2.z[indices]
        catalog2.w = catalog2.w[indices]
        catalog2.g1 = catalog2.g1[indices]
        catalog2.g2 = catalog2.g2[indices]

        data1 = np.column_stack([catalog1.ra,catalog1.dec,catalog1.z,catalog1.w,catalog1.g1,catalog1.g2])
        data2 = np.column_stack([catalog2.ra,catalog2.dec,catalog2.z,catalog2.w,catalog2.g1,catalog2.g2])

        return data1,data2


class RSD_2PCF:
    def __init__(self,*args,**qwargs):
        self.Om =  qwargs['Om0']
        self.mu_max = 1
        self.nmu_bins = 100
        self.nthreads = 4
        self.cosmo = FlatLambdaCDM(Om0= self.Om,H0=100)
        self.precompute_smu=0
        self.precompute_rppi=0
        self.normPCF = 0

        self.fileout1 = "RR_smu.dat" 
        self.fileout2 = "RR_rppi.dat"

        x = len(qwargs)
        if ( x > 1):
            self.ra =  qwargs['RA']
            self.dec =  qwargs['DEC']
            self.w =  qwargs['W'] 
            if 'Z' in qwargs:
                self.z =  qwargs['Z']
                self.dc = self.cosmo.comoving_distance(self.z).value
            else :
                self.dc = qwargs['Dc']

        elif (x > 5):
            self.ra_r =  qwargs['RA_r']
            self.dec_r =  qwargs['DEC_r']
            self.z_r =  qwargs['Z_r']
            self.w_r =  qwargs['W_r']  
            if 'Z_r' in qwargs:
                self.z_r =  qwargs['Z_r']
                self.dc_r = self.cosmo.comoving_distance(self.z_r).value
            else :
                self.dc_r = qwargs['Dc_r']

    def set_norm(self):
        self.normDD = np.sum(self.w)*np.sum(self.w) - np.sum(self.w**2)
        self.normRR = np.sum(self.w_r)*np.sum(self.w_r) - np.sum(self.w_r**2)
        self.normDR = np.sum(self.w)*np.sum(self.w_r)


    def set_random(self,*args,**qwargs):
        self.ra_r =  qwargs['RA_r']
        self.dec_r =  qwargs['DEC_r']
        self.w_r =  qwargs['W_r']        
        if 'Z_r' in qwargs:
            self.z_r =  qwargs['Z_r']
            self.dc_r = self.cosmo.comoving_distance(self.z_r).value
        else :
            self.dc_r = qwargs['Dc_r']
        
    def set_data(self,*args,**qwargs):

        self.ra =  qwargs['RA']
        self.dec =  qwargs['DEC']
        self.w =  qwargs['W']
        if 'Z' in qwargs:
            self.z =  qwargs['Z']
            self.dc = self.cosmo.comoving_distance(self.z).value
        else :
            self.dc = qwargs['Dc']

        
    def precompute_RR_smu(self,binsfile):
        self.precompute_smu = 1
        self.binsfile = binsfile
        self.ds = binsfile[1]-binsfile[0]
        self.centerbin = np.zeros(len(self.binsfile) - 1)
        self.mu = np.linspace(0,self.mu_max,self.nmu_bins + 1)

        self.mubins = np.zeros(len(self.mu) - 1)
        for i in range(0,len(self.mubins)):
            self.mubins[i] = (self.mu[i] + self.mu[i+1])/2
        du = self.mubins[1] - self.mubins[0]

        for i in range(0,len(self.centerbin)):
            self.centerbin[i] = (self.binsfile[i] + self.binsfile[i+1])/2

        autocorr=1        

        if os.path.exists(self.fileout1):
            RRpairs_smu,RRweights_smu = np.loadtxt(self.fileout1,unpack=True)
            self.RR = RRpairs_smu.reshape(len(self.centerbin),len(self.mubins))
            self.wRR = RRweights_smu.reshape(len(self.centerbin),len(self.mubins))
            self.precompute_smu = 1
        else : 
            self.RRpairs_smu = DDsmu_mocks(autocorr,1,self.nthreads,self.mu_max,self.nmu_bins,self.binsfile,
                    self.ra_r,self.dec_r,self.dc_r,weights1=self.w_r,
                    weight_type='pair_product',is_comoving_dist=True,output_savg=True)
            
            np.savetxt(self.fileout1,np.transpose([self.RRpairs_smu['npairs'],self.RRpairs_smu['weightavg']]))
            self.RR = self.RRpairs_smu['npairs'].reshape(len(self.centerbin),len(self.mubins))
            self.wRR = self.RRpairs_smu['weightavg'].reshape(len(self.centerbin),len(self.mubins))

        
    def precompute_RR_rppi(self,binsfile,pimax):
        self.pi_max=pimax
        self.precompute_rppi = 1
        self.binsfile = binsfile
        self.centerbin = np.zeros(len(self.binsfile) - 1)

        for i in range(0,len(self.centerbin)):
            self.centerbin[i] = (self.binsfile[i] + self.binsfile[i+1])/2


        autocorr=1        
        
        if os.path.exists(self.fileout2):
            RRpairs_rppi,RRweights_rppi = np.loadtxt(self.fileout2,unpack=True)
            self.RR_rppi = RRpairs_rppi.reshape(len(self.centerbin),self.pi_max)
            self.wRR_rppi = RRweights_rppi.reshape(len(self.centerbin),self.pi_max)
            self.precompute_rppi = 1
        else:
            self.RRpairs_rppi = DDrppi_mocks(autocorr,1,self.nthreads,self.pi_max,self.binsfile,
                        self.ra_r,self.dec_r,self.dc_r,weights1=self.w_r,
                        weight_type='pair_product',is_comoving_dist=True,output_rpavg=True)
            np.savetxt(self.fileout2,np.transpose([self.RR_rppi['npairs'],self.RR_rppi['weightavg']]))
            self.RR_rppi = self.RRpairs_rppi['npairs'].reshape(len(self.centerbin),self.pi_max)
            self.wRR_rppi = self.RRpairs_rppi['weightavg'].reshape(len(self.centerbin),self.pi_max)

        

    def compute(self,binsfile):
        if self.normPCF == 0:
            self.set_norm()
            self.normPCF == 1

        self.binsfile = binsfile
        self.ds = binsfile[1]-binsfile[0]
        self.centerbin = np.zeros(len(self.binsfile) - 1)
        self.mu = np.linspace(0,self.mu_max,self.nmu_bins + 1)

        self.mubins = np.zeros(len(self.mu) - 1)
        for i in range(0,len(self.mubins)):
            self.mubins[i] = (self.mu[i] + self.mu[i+1])/2
        du = self.mubins[1] - self.mubins[0]

        for i in range(0,len(self.centerbin)):
            self.centerbin[i] = (self.binsfile[i] + self.binsfile[i+1])/2

        autocorr=1

        self.DD = DDsmu_mocks(autocorr,1,self.nthreads,self.mu_max,self.nmu_bins,self.binsfile,
                        self.ra,self.dec,self.dc,weights1=self.w, weight_type='pair_product',is_comoving_dist=True,
                         output_savg=True)

        print('DD computation done')

        autocorr=0

        self.DR = DDsmu_mocks(autocorr,1,self.nthreads,self.mu_max,self.nmu_bins,self.binsfile,self.ra,self.dec,
                              self.dc, weights1 = self.w, RA2 = self.ra_r, DEC2=self.dec_r, CZ2=self.dc_r,
                            weight_type='pair_product',weights2=self.w_r,is_comoving_dist=True,output_savg=True)

        print('DR computation done')

        self.precompute_RR_smu(binsfile)


        DD = self.DD['npairs'].reshape(len(self.centerbin),len(self.mubins))
        wDD = self.DD['weightavg'].reshape(len(self.centerbin),len(self.mubins))

        DR = self.DR['npairs'].reshape(len(self.centerbin),len(self.mubins))
        wDR = self.DR['weightavg'].reshape(len(self.centerbin),len(self.mubins))

        self.xismu = (DD*wDD)/(self.RR*self.wRR)*(self.normRR/self.normDD) -2*(DR*wDR)/(self.RR*self.wRR)*(self.normRR/self.normDR) + 1

        e0 = np.zeros((len(self.centerbin),len(self.mubins)))
        e2 = np.zeros((len(self.centerbin),len(self.mubins)))
        e4 = np.zeros((len(self.centerbin),len(self.mubins)))

        for i in range(0,len(self.xismu[:,0])):
            e0[i] = (1./2.)*self.xismu[i]
            e2[i] = (5./2.)*self.xismu[i]*L2(self.mubins)
            e4[i] = (9./2.)*self.xismu[i]*L4(self.mubins)

        e0 = np.sum(e0*du,axis=1)*2.
        e2 = np.sum(e2*du,axis=1)*2.
        e4 = np.sum(e4*du,axis=1)*2.

        return self.centerbin,e0,e2,e4
    

    
    def compute_wp(self,binsfile,pimax):
        if self.normPCF == 0:
            self.set_norm()
            self.normPCF == 1
        
        self.binsfile = binsfile
        self.pi_max = pimax
        self.centerbin = np.zeros(len(self.binsfile) - 1)
        
        for i in range(0,len(self.centerbin)):
            self.centerbin[i] = (self.binsfile[i] + self.binsfile[i+1])/2
            
        
        autocorr=1

        self.DD = DDrppi_mocks(autocorr,1,self.nthreads,self.pi_max,self.binsfile,
                        self.ra,self.dec,self.dc,weights1=self.w,\
                               weight_type='pair_product',is_comoving_dist=True,
                         output_rpavg=True)

        print('DD computation done')

        autocorr=0

        self.DR = DDrppi_mocks(autocorr,1,self.nthreads,self.pi_max,self.binsfile,self.ra,self.dec,
                              self.dc, weights1 = self.w, RA2 = self.ra_r, DEC2=self.dec_r,\
                              CZ2=self.dc_r,weight_type='pair_product',weights2=self.w_r,\
                              is_comoving_dist=True,output_rpavg=True)

        self.precompute_RR_rppi(binsfile,self.pi_max)

        print('DR computation done')


        DD = self.DD['npairs'].reshape(len(self.centerbin),self.pi_max)
        wDD = self.DD['weightavg'].reshape(len(self.centerbin),self.pi_max)

        DR = self.DR['npairs'].reshape(len(self.centerbin),self.pi_max)
        wDR = self.DR['weightavg'].reshape(len(self.centerbin),self.pi_max)


        self.xirppi = (DD*wDD)/(self.RR_rppi*self.wRR_rppi)*(self.normRR/self.normDD) -2*(DR*wDR)/(self.RR_rppi*self.wRR_rppi)*(self.normRR/self.normDR) + 1

        wp = 2*np.sum(self.xirppi,axis=1)
            
        return self.centerbin,wp
    


#class Real_2PCF:
#    def __init__
