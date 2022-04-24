import sys
import os
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.io import fits,ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from scipy.interpolate import CubicSpline as CS
import HOD_model
import PCF
pd.options.mode.chained_assignment = None
from scipy.constants import speed_of_light
from iminuit import Minuit 
import emcee
from multiprocessing import Pool
import os.path 
#import os
#os.environ["OMP_NUM_THREADS"] = "1"

c = speed_of_light/1e3


rng = np.random.default_rng()


class HOD_FIT:
    def __init__(self,path,tracer,datapath,covpath):
        self.tracer = tracer

        if tracer == "ELG":
            self.zmin = 0.6
            self.zmax = 1.1
            self.Mcut = 2e11
        elif tracer == "LRG":
            self.zmin = 0.6
            self.zmax = 1.1
            self.Mcut = 5e11
        elif tracer == "CMASS":
            self.zmin = 0.43
            self.zmax = 0.7
            self.Mcut = 5e11

        x = np.loadtxt(datapath)
        self.rp,self.wpdata,_ = np.loadtxt(datapath,unpack=True)
        self.cov = np.loadtxt(covpath,unpack=True)
        self.invcov = np.linalg.inv(self.cov)*(100 - len(self.rp))/(100 - 1.)
        
        T = Table.read(path)
        df = T.to_pandas()
        df=df[(df.M200C > 2e11)&(df.z_obs > self.zmin-0.01) & (df.z_obs < self.zmax+0.01)]

        z = df['z_obs'].values
        ra = df['RA'].values
        dec = df['DEC'].values
        w = np.ones(len(dec))
        
        """Create random catalog"""
        file = "random_uchuu_footprint.dat"
        if os.path.exists(file):
            ra_r,dec_r,z_r,w_r = np.loadtxt(file,unpack=True)
            ra_r = self.convert_ra(ra_r)
        else:
            rr = PCF.randomcat(ra,dec,z,w) 
            ra_r,dec_r,z_r,w_r = rr.create_random_easy(3e6)
            cond = (z_r > self.zmin) & (z_r < self.zmax)
            ra_r = ra_r[cond]
            dec_r = dec_r[cond]
            z_r = z_r[cond] 
            w_r = w_r[cond]
            np.savetxt(file,np.transpose([ra_r,dec_r,z_r,w_r]))
            ra_r = self.convert_ra(ra_r)

#        df.RA = self.convert_ra(df.RA)
        df = df[(df.M200C > 2e11) & (df.z_obs > self.zmin) & (df.z_obs < self.zmax)]
        df = df.reset_index(drop=True)
        dfm = df[df.ID == df.IDp]
        dft = df[df.ID != df.IDp]
        self.dft = dft.reset_index(drop=True)
        self.dfm = dfm.reset_index(drop=True)



        """Logarithmic binning for rp"""
        self.rbins = np.geomspace(0.2,50,13)
        self.pimax = 80
        self.pipeline_2PCF = PCF.RSD_2PCF(Om0=0.3089)
        self.pipeline_2PCF.set_random(RA_r=ra_r,DEC_r=dec_r,Z_r=z_r,W_r=w_r)
        ##self.pipeline_2PCF.precompute_RR_smu(self.rbins)
        self.pipeline_2PCF.precompute_RR_rppi(self.rbins,self.pimax)
        self.HOD = HOD_model.HOD(self.Mcut)

    @staticmethod
    def subsample(x,y):
        x = x[0:y]
        return x


    def convert_ra(self,ra):
        cond = np.where(ra < 0)
        ra[cond]=ra[cond]+360
        return ra
    
    def populate(self,p):
        self.dfgC = pd.DataFrame(columns=['IDp','ID','RA','DEC','X','Y','Z','Rcom','Rcomc','VX','VY','VZ','M200P','M200C','z_true','z_obs','Rcom_obs'])
        self.dfgS = pd.DataFrame(columns=['IDp','ID','RA','DEC','X','Y','Z','Rcom','Rcomc','VX','VY','VZ','M200P','M200C','z_true','z_obs','Rcom_obs'])
        self.dfg = pd.DataFrame(columns=['IDp','ID','RA','DEC','X','Y','Z','Rcom','Rcomc','VX','VY','VZ','M200P','M200C','z_true','z_obs','Rcom_obs'])

        self.HOD.set_params(self.tracer,p)
    
        """Centrals"""

        probC = self.HOD.splineC(self.dfm.M200C)
        probC[probC > 1] = 1
        array = np.ones(probC.size,dtype=int)
        Nc = rng.binomial(array,probC)
        cond = Nc > 0
        self.dfm2 = self.dfm[cond]
        self.dfgC=pd.concat([self.dfgC,self.dfm2],axis=0)

        """Sattelites"""

        self.group_sub = self.dft.groupby(['IDp','Rcomc','M200P'],as_index=False)
        self.group_info = self.group_sub.size()
        M200P = self.group_info['M200P'].values
        probS = self.HOD.splineS(M200P)
        Ns = rng.poisson(probS)
        indices = self.group_sub.indices
        indexes = np.array(list(indices.values()),dtype="object")
        cond = np.where(Ns > 0)
        M200P = M200P[cond]
        Ns = Ns[cond]
        indexes = indexes[cond]
        list(map(np.random.shuffle, indexes))
        indexes2 = np.array(list(map(HOD_FIT.subsample, indexes,Ns)),dtype="object")
        indextot = np.hstack(indexes2)
        self.dfgS = pd.concat([self.dfgS,self.dft.loc[indextot]])
        self.dfg  = pd.concat([self.dfgC,self.dfgS])


    def galaxy_wp(self):
        ra = (self.dfg['RA'].values).astype('<f8')
        dec = (self.dfg['DEC'].values).astype('<f8')
        Dc = (self.dfg['Rcom_obs'].values).astype('<f8')
        #z = (self.dfg['z_obs'].values).astype('<f8')
        w = np.ones(len(ra)).astype('<f8')
        ra = self.convert_ra(ra)
        self.pipeline_2PCF.set_data(RA=ra,DEC=dec,Dc=Dc,W=w)
        #r,e0,e2,e4 = self.pipeline_2PCF.compute(self.rbins)
        rp,wp = self.pipeline_2PCF.compute_wp(self.rbins,self.pimax)
        self.rp = rp
        self.wp = wp
        #self.clustering = np.column_stack([wp,e2])

        return self.rp,self.wp

    def galaxy_wp_central(self):
        ra = (self.dfgC['RA'].values).astype('<f8')
        dec = (self.dfgC['DEC'].values).astype('<f8')
        Dc = (self.dfgC['Rcom_obs'].values).astype('<f8')
        #z = (self.dfg['z_obs'].values).astype('<f8')
        w = np.ones(len(ra)).astype('<f8')
        ra = self.convert_ra(ra)
        self.pipeline_2PCF.set_data(RA=ra,DEC=dec,Dc=Dc,W=w)
        #r,e0,e2,e4 = self.pipeline_2PCF.compute(self.rbins)
        rp,wp = self.pipeline_2PCF.compute_wp(self.rbins,self.pimax)
        self.rp = rp
        self.wp = wp
        #self.clustering = np.column_stack([wp,e2])

        return self.rp,self.wp


    def compute_multipoles(self,binsfile):
        ra = (self.dfg['RA'].values).astype('<f8')
        dec = (self.dfg['DEC'].values).astype('<f8')
        z = (self.dfg['z_obs'].values).astype('<f8')
        w = np.ones(len(ra)).astype('<f8')
        ra = self.convert_ra(ra)
        self.pipeline_2PCF.set_data(RA=ra,DEC=dec,Z=z,W=w)
        s,e0,e2,e4 = self.pipeline_2PCF.compute(binsfile)
        return s,e0,e2,e4

    def compute_multipoles_central(self,binsfile):
        ra = (self.dfgC['RA'].values).astype('<f8')
        dec = (self.dfgC['DEC'].values).astype('<f8')
        z = (self.dfgC['z_obs'].values).astype('<f8')
        w = np.ones(len(ra)).astype('<f8')
        ra = self.convert_ra(ra)
        self.pipeline_2PCF.set_data(RA=ra,DEC=dec,Z=z,W=w)
        s,e0,e2,e4 = self.pipeline_2PCF.compute(binsfile)
        return s,e0,e2,e4


    def fsat(self):
        fsat = self.dfgS.size/(self.dfgS.size+self.dfgC.size)
        return fsat


    def fit(self):
        if self.tracer == "ELG":
            self.ELG_fit()
            #return self.pmax,self.emax
        else :
            self.LRG_fit()
            return self.pmax,self.emax

    def ELG_clustering(self,p):
        self.populate(p)
        self.rp,self.wp = self.galaxy_wp()


    def LRG_clustering(self,p):
        self.populate(p)
        self.rp,self.wp = self.galaxy_wp()


    def chi2_ELG(self,Ac,As,mu):
        p = np.array([Ac,As,mu])
        self.ELG_clustering(p)
        diff = self.wp - self.wpdata
        chi = np.dot(diff,np.dot(self.invcov,diff))
        return chi

    def logL_ELG(self,p):
        Ac,As,mu = p
        chi = self.chi2_ELG(Ac,As,mu)
        return -0.5*chi

    def prior(self,p):
        Ac,As,mu = p
        if 0.004 < Ac < 0.007 and 0.004 < As < 0.007 and 11.2 < mu < 11.8:
            return 0.0
        return -np.inf

    def lnprob_ELG(self,p):
        lp = self.prior(p)
        like = 0
        if not np.isfinite(lp):
            return -np.inf
        else :
            like = self.logL_ELG(p)
            return lp + like



    def chi2_LRG(self,Mmin,M0,M1,sigma,alpha):
        p = np.array([Mmin,M0,M1,sigma,alpha])
        self.LRG_clustering(p)
        diff = self.wp - self.wpdata
        chi = np.dot(diff,np.dot(self.invcov,diff))
        return chi

    def ELG_fit(self):
        m = Minuit(self.chi2_ELG,Ac = 0.00537,As = 0.005301,mu = 11.7)
        m.errordef = 1
        m.limits = [(0.004,0.007), (0.004,0.007),(11,13)]
        m.errors['Ac'] = 0.0002
        m.errors['As'] = 0.0002
        m.errors['mu'] = 0.01
        m.migrad()
        xi2 = m.fval
        fmin2 = m.fmin
        self.xi2 = xi2

        self.Ac=m.values['Ac']
        self.As=m.values['As']
        self.mu = m.values['mu']

        self.pmax = np.hstack([self.Ac,self.As,self.mu])

        self.eAc=m.errors['Ac']
        self.eAs=m.errors['As']
        self.emu = m.errors['mu']

        self.emax = np.hstack([self.eAc,self.eAs,self.emu])


    def ELG_fit_emcee(self):
        pos = np.array([0.0053,0.0053,11.5]) + (5e-4,5e-4,2e-1) * np.random.randn(20, 3)
        nwalkers, ndim = pos.shape

        nsteps = 3000

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob_ELG)
            sampler.run_mcmc(pos, nsteps, progress=True)

        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        d = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
        np.savetxt("/feynman/work/dap/lceg/rp261901/stuff/chain_fit_HOD_ELG.dat",d)  

#Mmin,M0,M1,sigma,alpha

    def LRG_fit(self):
        m = Minuit(self.chi2_LRG,Ac = 0.00537,As = 0.005301,mu = 11.7,error_Ac=0.0001,error_As=0.0001\
            ,errordef=1)


        m.migrad()
        xi2 = m.fval
        fmin2 = m.fmin
        self.xi2 = xi2

        self.Ac=m.values['Ac']
        self.As=m.values['As']
        self.mu = m.values['mu']

        self.pmax = np.hstack([self.Ac,self.As,self.mu])

        self.eAc=m.errors['Ac']
        self.eAs=m.errors['As']
        self.emu = m.errors['mu']

        self.emax = np.hstack([self.eAc,self.eAs,self.emu])

