import sys
##sys.path.append('/feynman/home/dap/lceg/rp269101/.local/lib/python3.6/site-packages')
##sys.path.append('/usr/lib64/python3.6/site-packages')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from random import randint
from astropy.io import fits,ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
#from scipy.special import erfc,erf
from mpmath import erf


## CMASS parameter
logMmin = 13.08
logM1 = 14.00
logM0 = 13.077
sigmalog = 0.596
alpha = 1.0127
Mmin = 10**logMmin
M1 = 10**logM1
M0 = 10**logM0

#ELG parameter
Ac = 0.00537
As = 0.005301
mu = 11.515
logM0e = mu - 0.05
logM1e = mu + 0.35
M0e = 10**logM0e
M1e = 10**logM1e
sigmae = 0.08
alphae = 0.9
gammae = -1.4

def HOD_centrals(M):
    Ncen = (1./2.)*(1 + erf(np.log(M/Mmin)/sigmalog))
    return Ncen

def HOD_sattelite(M):
    if (M > M0):
        Ncen = HOD_centrals(M)
        Nsat=Ncen*((M - M0)/M1)**(alpha)
    else :
        Nsat=0.0
    return Nsat

def HOD_central_ELG(M):
    x = np.log10(M)
    if (x < mu):
        Ncen = Ac/(np.sqrt(2*np.pi)*sigmae)*np.exp(-((x - mu)**2)/2*sigmae**2)
    else:
        Ncen= Ac/(np.sqrt(2*np.pi)*sigmae)*(M/10**mu)**(gammae)

    return Ncen

def HOD_satelite_ELG(M):
    Nsat = As*((M - M0e/M1e)**(alphae)



def random_sample(side):
    return np.random.uniform(side)




HOD_sattelitev = np.vectorize(HOD_sattelite)
HOD_sattelite_ELGv = np.vectorize(HOD_sattelite_ELG)
HOD_central_ELGv = np.vectorize(HOD_central_ELG)




def load_data(filename):
    df = dd.read_parquet(filename)
    return df


df = load_data('/feynman/work/dap/lceg/rp269101/lightcone2_wide')
df = df[df.M200C > 3e10]
dfm= df[df.PID == -1]    
dfs = df[df.PID != -1]
dfm2 = dfm.copy()
dfm2 = dfm2.drop(columns=['RA','DEC','Rcom','RS','X','Y','Z','VX','VY','VZ','VMAX','PID','RVIR'])
dfm2 = dfm2.rename(columns={"M200C": "M200CC", "ID":"IDt"}) 
dfs = dfs.rename(columns={"PID":"IDt"})
dft = dd.merge(dfm2,dfs,on="IDt")
dfm = dfm.drop(columns={"ID","RS","PID","X","Y","Z"})
dft = dft.drop(columns={"IDt","RS","ID","X","Y","Z"})

dfm = dfm.compute()
dft = dft.compute()
Mh_u = dft.M200CC.unique()
#dds = drop_duplicates
#probC = dask.delayed(HOD_centrals)(dfm.M200C)
#probS = dask.delayed(HOD_sattelitev)(Mh_u)

probC = dask.delayed(HOD_central_ELGv)(dfm.M200C)
probS = dask.delayed(HOD_sattelite_ELGv)(Mh_u)


array = dask.delayed(np.ones)(probC.size,dtype=int)
Nc = dask.delayed(np.random.binomial)(array,probC)
Ns = dask.delayed(np.random.poisson)(probS)

    
Nc = Nc.compute()
Ns = Ns.compute()



dfg = pd.DataFrame(columns=['RA','DEC','Rcom','VX','VY','VZ','VMAX','M200C','RVIR'])

#t1 = time.time()
for i in range(0,Mh_u.size):
    dfi = dft.loc[dft.M200CC == Mh_u[i]]
    size = dfi.M200CC.size
    dfi = dfi.drop(columns={"M200CC"})
    if Ns[i] < size : 
        indices = np.random.choice(np.linspace(0,size-1,size),size=Ns[i], replace=False)  
        indices = list(indices)
        dfi = dfi.iloc[indices]
        dfg = pd.concat([dfg, dfi])
    else :
        dfg = pd.concat([dfg,dfi])
#t2 = time.time()

dfgC = dfm[Nc == 1]
dfg = pd.concat([dfg, dfgC])    


z = z_at_value(cosmo.comoving_distance,dfg.Rcom.values*u.Mpc)
dfg['z'] = z.value
dfg = dfg.drop(columns={'Rcom'})
t = Table.from_pandas(dfg.astype("float64"))
t.write('galaxy_catalog_ELG_uchuu.fits', format='fits',overwrite=True)
