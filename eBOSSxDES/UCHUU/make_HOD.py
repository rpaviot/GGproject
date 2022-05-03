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
pd.options.mode.chained_assignment = None
from scipy.constants import speed_of_light
from iminuit import Minuit 
import emcee
from multiprocessing import Pool
import os.path 

import sys
sys.path.insert(0,"../tools")
import PCF
from geo import survey_geometry


"""Main class that perform the HOD fit on UCHUU mocks"""

c = speed_of_light/1e3


rng = np.random.default_rng()


class HOD_FIT:
    def __init__(self,path,tracer,datapath,covpath):
        self.tracer = tracer
        self.Om0=0.3089
        self.cosmo = FlatLambdaCDM(Om0=self.Om0,H0=100)
        if tracer == "ELG":
            self.zmin = 0.6
            self.zmax = 1.1
            self.dcmin = self.cosmo.comoving_distance(self.zmin).value
            self.dcmax = self.cosmo.comoving_distance(self.zmax).value
            self.Mcut = 2e11
        elif tracer == "LRG":
            self.zmin = 0.6
            self.zmax = 1.0
            self.Mcut = 5e11
            self.dcmin = self.cosmo.comoving_distance(self.zmin).value
            self.dcmax = self.cosmo.comoving_distance(self.zmax).value
        elif tracer == "CMASS":
            self.zmin = 0.43
            self.zmax = 0.7
            self.Mcut = 5e11
            self.dcmin = self.cosmo.comoving_distance(self.zmin).value
            self.dcmax = self.cosmo.comoving_distance(self.zmax).value

        self.set_data(self.tracer)
        self.rp,self.wpdata,_ = np.loadtxt(datapath,unpack=True)
        self.cov = np.loadtxt(covpath,unpack=True)
        self.invcov = np.linalg.inv(self.cov)*(100 - len(self.rp))/(100 - 1.)

        T = Table.read(path)
        df = T.to_pandas()
        df=df[(df.M200C > 2e11)&(df.M200P > 2e11)&(df.Rcom_obs > self.dcmin) & (df.Rcom_obs < self.dcmax)]
        dfm = df[df.ID == df.IDp]
        dft = df[df.ID != df.IDp]
        self.dft = dft.reset_index(drop=True)
        self.dfm = dfm.reset_index(drop=True)

        ra = self.dfm['RA'].values
        dec = self.dfm['DEC'].values
        Dc = self.dfm['Rcom_obs'].values
        w = np.ones(len(Dc))
        
        self.survey = survey_geometry(NSIDE=512,RA=ra,DEC=dec,Dc=Dc,W=w) 
        self.survey.set_target_nr(self.dc_data,self.w_data)

        """Logarithmic binning for rp"""
        self.rbins = np.geomspace(0.2,50,13)
        self.pimax = 80
        self.HOD = HOD_model.HOD(self.Mcut)

        self.init_RR(self.tracer)




        #self.dft = self.dft.reset_index(drop=True)
        #self.dfm = self.dfm.reset_index(drop=True)
 
        del df 


    @staticmethod
    def subsample(x,y):
        x = x[0:y]
        return x


    def convert_ra(self,ra):
        cond = np.where(ra < 0)
        ra[cond]=ra[cond]+360
        return ra


    def set_data(self,tracer):
        if tracer == "ELG":
            pathdata="/Users/rpaviot/eBOSSxDES/faizan/elg/eBOSS_ELG_SGC_pip_v7.dat.fits"
            T = Table.read(pathdata)
            cond = ((T['CLUSTERING']==1) & (T['Z'] > self.zmin) & (T['Z'] < self.zmax))
            T = T[cond]
        else:
            pathdata="/Users/rpaviot/Downloads/eBOSS_LRGpCMASS_clustering_data-SGC-vDR16.fits"
            T = Table.read(pathdata)
            cond =  ((T['Z'] > self.zmin) & (T['Z'] < self.zmax))
            T = T[cond]
        self.z_data = T['Z'].value
        self.dc_data = self.cosmo.comoving_distance(self.z_data).value
        self.w_data = np.ones(len(self.z_data))


    """Create random catalog"""

    def init_RR(self,tracer):
        file = "random_uchuu_{}_footprint.dat".format(tracer)
        if os.path.exists(file):
            ra_r,dec_r,dc_r,w_r = np.loadtxt(file,unpack=True)
            ra_r = self.convert_ra(ra_r)
        else:
            ra_r,dec_r,dc_r,w_r = self.survey.create_random_easy(2e6)
            cond = (dc_r > self.dcmin) & (dc_r < self.dcmax)
            ra_r = ra_r[cond]
            dec_r = dec_r[cond]
            dc_r = dc_r[cond] 
            w_r = w_r[cond]
            np.savetxt(file,np.transpose([ra_r,dec_r,dc_r,w_r]))
            ra_r = self.convert_ra(ra_r)

        self.pipeline_2PCF = PCF.RSD_2PCF(Om0=self.Om0,RR_file=file)
        self.pipeline_2PCF.set_random(RA_r=ra_r,DEC_r=dec_r,Dc_r=dc_r,W_r=w_r)
        self.pipeline_2PCF.precompute_RR_rppi(self.rbins,self.pimax)


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
        cond = Ns > 0
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
        self.ra_mock,self.dec_mock,self.Dc_mock,self.w_mock = self.survey.set_mock_nz(ra,dec,Dc,w)
        self.pipeline_2PCF.set_data(RA=self.ra_mock,DEC=self.dec_mock,Dc=self.Dc_mock,W=self.w_mock)
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
        self.ra_mock,self.dec_mock,self.Dc_mock,self.w_mock = self.survey.set_mock_nz(ra,dec,Dc,w)
        self.pipeline_2PCF.set_data(RA=self.ra_mock,DEC=self.dec_mock,Dc=self.Dc_mock,W=self.w_mock)
        #r,e0,e2,e4 = self.pipeline_2PCF.compute(self.rbins)
        rp,wp = self.pipeline_2PCF.compute_wp(self.rbins,self.pimax)
        self.rp = rp
        self.wp = wp
        #self.clustering = np.column_stack([wp,e2])

        return self.rp,self.wp


    def compute_multipoles(self,binsfile):
        ra = (self.dfg['RA'].values).astype('<f8')
        dec = (self.dfg['DEC'].values).astype('<f8')
        Dc = (self.dfg['Rcom_obs'].values).astype('<f8')
        w = np.ones(len(ra)).astype('<f8')
        ra = self.convert_ra(ra)
        self.ra_mock,self.dec_mock,self.Dc_mock,self.w_mock = self.survey.set_mock_nz(ra,dec,Dc,w)
        self.pipeline_2PCF.set_data(RA=self.ra_mock,DEC=self.dec_mock,Dc=self.Dc_mock,W=self.w_mock)
        s,e0,e2,e4 = self.pipeline_2PCF.compute(binsfile)
        return s,e0,e2,e4

    def compute_multipoles_central(self,binsfile):
        ra = (self.dfgC['RA'].values).astype('<f8')
        dec = (self.dfgC['DEC'].values).astype('<f8')
        Dc = (self.dfgC['Rcom_obs'].values).astype('<f8')
        w = np.ones(len(ra)).astype('<f8')
        ra = self.convert_ra(ra)
        self.ra_mock,self.dec_mock,self.Dc_mock,self.w_mock = self.survey.set_mock_nz(ra,dec,Dc,w)
        self.pipeline_2PCF.set_data(RA=self.ra_mock,DEC=self.dec_mock,Dc=self.Dc_mock,W=self.w_mock)
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
        log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
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
