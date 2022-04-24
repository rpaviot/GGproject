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
c = speed_of_light/1e3

class clustering_catalog:

    def __init__(self,path):
        """Create central and sattelite catalogs within redshift range 0.43 < z < 1.1"""
        cosmo = FlatLambdaCDM(Om0=0.3089,H0=100)
        zmin = 0.42
        zmax = 1.11
        self.Dmin = cosmo.comoving_distance(zmin)
        self.Dmax = cosmo.comoving_distance(zmax)
        z = np.arange(zmin,zmax,0.005)
        Dc = cosmo.comoving_distance(z)
        """inverse spline to get z given Dc"""
        self.spline_z = CS(Dc,z)
        
        self.dfC = pd.DataFrame(columns=['IDp','ID','RA','DEC','Rcom','Rcomc','VX','VY','VZ','M200P','M200C'])
        self.dfS = pd.DataFrame(columns=['IDp','ID','RA','DEC','Rcom','Rcomc','VX','VY','VZ','M200P','M200C'])

        self.path = path
        files = os.listdir(self.path)
        
        for i in range(0,len(files)):
            pathfile = self.path + files[i]
            self.read_catalog(pathfile)
            if self.df.size == 0:
                continue
            self.parent_catalog()
            self.child_catalog()
            self.match_catalog()
            
        self.dftt = pd.concat([self.dfC,self.dfS])
        self.dftt = self.dftt.reset_index(drop=True)
        self.dftt['z_true'] = self.spline_z(self.dftt['Rcom'])
        xc = self.dftt['Rcom']*np.cos(self.dftt['DEC']*np.pi/180)*np.cos(self.dftt['RA']*np.pi/180)
        yc = self.dftt['Rcom']*np.cos(self.dftt['DEC']*np.pi/180)*np.sin(self.dftt['RA']*np.pi/180)
        zc = self.dftt['Rcom']*np.sin(self.dftt['DEC']*np.pi/180)
        r = np.sqrt(xc**2 + yc**2 + zc**2)
        vr = (xc*self.dftt['VX'] + yc*self.dftt['VY'] + zc*self.dftt['VZ'])/r
        dz = (1+self.dftt['z_true'])*vr/c
        zrsd = self.dftt['z_true'] + dz
        self.dftt['z_obs'] = zrsd
        #self.dftt['Rcom_obs']= self.dftt['Rcom'] + vr*(1+dftt['z_true'])/cosmo.H(dftt['z_true']).value
        self.savecatalog("/feynman/work/dap/lceg/rp269101/stuff/")

    def read_catalog(self,path):
        self.df = pd.read_parquet(path)
        self.df = self.df[(self.df.M200C > 1e11)&(self.df.Rcom > self.Dmin)&(self.df.Rcom < self.Dmax)]

    def parent_catalog(self):
        self.dfm = self.df[self.df.PID == -1]
        self.dfm = self.dfm.drop(columns=['PID','RS','RVIR','VMAX','X','Y','Z'])
        self.dfm['IDp'] = self.dfm['ID']
        self.dfm['M200P'] = self.dfm['M200C']
        self.dfm['Rcomc'] = self.dfm['Rcom']
        self.dfm = self.dfm[['IDp','ID','RA','DEC','Rcom','Rcomc','VX','VY','VZ','M200P','M200C']]
        self.dfC = pd.concat([self.dfC,self.dfm])
        del self.dfm

    def child_catalog(self):
        self.dfs = self.df[self.df.PID != -1]
        self.dfs = self.dfs.rename(columns={"PID":"IDp"})
        self.dfs = self.dfs.drop(columns=['RS','RVIR','VMAX','X','Y','Z'])
        
    def match_catalog(self):
        """ Merge to get for each subhalo the mass of the parent halo"""
        dfm2 = self.dfm.copy()
        dfm2 = dfm2.drop(columns=['ID','M200C','RA','DEC','Rcom','VX','VY','VZ'])
        self.dft = pd.merge(dfm2,self.dfs,on="IDp")
        self.dft = self.dft[['IDp','ID','RA','DEC','Rcom','Rcomc','VX','VY','VZ','M200P','M200C']]
        self.dfS = pd.concat([self.dfS,self.dft])
        del dfm2
        del self.dft
        del self.df
    def savecatalog(self,path):
        t = Table.from_pandas()
        t.write(path + 'clustering_catalog_uchuu.fits', format='fits',overwrite=True)