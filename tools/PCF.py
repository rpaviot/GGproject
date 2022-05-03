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
from scipy.interpolate import CubicSpline as CS
import os.path


"""Code to compute RSD clustering with Corrfunc"""


L0 = lambda x : 1
L2 =  lambda x : (1./2)*(3*x**2-1)
L4 = lambda x : (35.0*x*x*x*x-30.0*x*x+3.0)/8.0
L6 = lambda x : (231*pow(x,6)-315*pow(x,4)+105*pow(x,2)-5)/16.0
L8 = lambda x : (6435*pow(x,8)-12012*pow(x,6) + 6930*pow(x,4) - 1260*pow(x,2) + 35)/128

from numpy.random import default_rng
rng = default_rng()


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

        if 'RR_file' in qwargs:
            self.fileout1 = 'RR_smu'+ qwargs['RR_file']
            self.fileout2 = 'RR_rppi'+ qwargs['RR_file'] 



        x = len(qwargs)
        if ( x > 2):
            self.ra =  qwargs['RA']
            self.dec =  qwargs['DEC']
            self.w =  qwargs['W'] 
            if 'Z' in qwargs:
                self.z =  qwargs['Z']
                self.dc = self.cosmo.comoving_distance(self.z).value
            else :
                self.dc = qwargs['Dc']

        elif (x > 6):
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
            np.savetxt(self.fileout2,np.transpose([self.RRpairs_rppi['npairs'],self.RRpairs_rppi['weightavg']]))
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