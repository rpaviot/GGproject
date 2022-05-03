import numpy as np
from astropy.io import fits,ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
import itertools
import pandas as pd
from scipy.stats import binned_statistic,binned_statistic_2d
import numpy_indexed as npi


cosmo = FlatLambdaCDM(Om0=0.3089,H0=100)

pathELG="/feynman/work/dap/lceg/rp269101/stuff/faizan/elg/eBOSS_ELG_SGC_pip_v7.dat.fits"
pathELGr="/feynman/work/dap/lceg/rp269101/stuff/faizan/elg/eBOSS_ELG_SGC_pip_v7.rnd.fits"

pathLRG="/feynman/work/dap/lceg/rp269101/stuff/faizan/lrg/eBOSS_LRG_SGC_pip_v7_2.dat.fits"
pathLRGr="/feynman/work/dap/lceg/rp269101/stuff/faizan/lrg/eBOSS_LRG_SGC_pip_v7_2.ran.fits"


hdul = fits.open(pathELG)
T = hdul[1].data

#T = Table.read(pathLRG)
"""Parent Catalog"""
rap = T['RA']
decp = T['DEC']
zp = T['Z']
wp = T['WEIGHT_SYSTOT']*T['WEIGHT_FKP']*T['WEIGHT_NOZ']
Dcp = cosmo.comoving_distance(zp).value
#wcpp = T['WEIGHT_CP']


"""Fiber catalog"""
cond = (T['FIBER']==1)
Tf= T[cond]
raf = Tf['RA']
decf = Tf['DEC']
zf = Tf['Z']
wf = Tf['WEIGHT_SYSTOT']*Tf['WEIGHT_FKP']*Tf['WEIGHT_NOZ']
wcpf = Tf['WEIGHT_CP']
Dcf = cosmo.comoving_distance(zp).value
W_BWf = Tf['WEIGHT_BW']
Nbitsf = int(W_BWf[0][0]).bit_length()*len(W_BWf[0])

"""Clustering catalog"""
cond = ((T['CLUSTERING']==1) & (T['Z'] > 0.6) & (T['Z'] < 1.1))
#print(T['RA'].value)
Tc = T[cond]
#Tc = Tc[0:1000]
ra = Tc['RA']
dec = Tc['DEC']
z = Tc['Z']
w = Tc['WEIGHT_SYSTOT']*Tc['WEIGHT_FKP']*Tc['WEIGHT_NOZ']
wcp = Tc['WEIGHT_CP']
Dc = cosmo.comoving_distance(z).value

"""WIIP apply only on DR pairs"""

W_BW = Tc['WEIGHT_BW']
Nbits = int(W_BW[0][0]).bit_length()*len(W_BW[0])


T2 = Table.read(pathELGr)
T2 = T2.to_pandas()
cond = ((T2['Z'] > 0.6) & (T2['Z'] < 1.1))
#cond = ((T2['CLUSTERING']==1) & (T2['Z'] > 0.6) & (T2['Z'] < 1.1))
T2 = T2[cond]
T2= T2.sample(frac=0.4)
ra_r = T2['RA'].values
dec_r = T2['DEC'].values
z_r = T2['Z'].values
w_r = (T2['WEIGHT_FKP']*T2['WEIGHT_NOZ']*T2['WEIGHT_CP']).values
wcp_r = T2['WEIGHT_CP'].values
Dc_r = cosmo.comoving_distance(z_r).value
del T2
    

def IIP(x):
    count = (x & x)
    Nn = np.unpackbits(count.view('uint8'))
    return Nbitsf/np.sum(Nn)

def PIP(x,y):
    count = (x & y)
    Nn = np.unpackbits(count.view('uint8'))
    return Nbitsf/np.sum(Nn)


wIIP= np.array(list(map(IIP,W_BW)))
wIIPf= np.array(list(map(IIP,W_BWf)))


def distance(cent,pos):
    dist = np.sqrt((pos[:,0] - cent[0])**2+(pos[:,1] - cent[1])**2+(pos[:,2] - cent[2])**2)
    return dist
    
def norm(vec):
    return np.sqrt(vec[:,0]**2+vec[:,1]**2+vec[:,2]**2)

def normr(vec):
    return np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)



def DD_brute_theta(*args,**qwargs):
    ra = args[0]
    dec = args[1]
    w =args[2]
    binsfile = args[3]

    x = np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180)
    y = np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180)
    z = np.sin(dec*np.pi/180)
    r = np.column_stack([x,y,z])
    r = r/norm(r)[:,None]
    size = x.size
        
    count_theta = np.zeros(len(binsfile)-1)
    normDD = 0
    
    for i in range(0,size-1):
        cent,wcent = r[i],w[i]
        wi = w[int(i+1):size]
        ri = r[int(i+1):size]
        mu = np.array([np.dot(cent,x) for x in ri])
        theta = np.arccos(mu)*180/np.pi
        hist,_,_= binned_statistic(theta,wi*wcent,statistic='sum',bins=binsfile)
        count_theta += hist
        normDD += np.sum(wi*wcent)
        
    return normDD,count_theta



def DD_brute_theta_PIP(*args,**qwargs):
    ra = args[0]
    dec = args[1]
    w =args[2]
    binsfile = args[3]
    W_BW = args[4]
    x = np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180)
    y = np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180)
    z = np.sin(dec*np.pi/180)
    r = np.column_stack([x,y,z])
    r = r/norm(r)[:,None]
    size = x.size
        
    count_theta = np.zeros(len(binsfile)-1)
    normDD = 0


    for i in range(0,size-1):
        cent,wcent,W_BWp = r[i],w[i],W_BW[i]
        wi = w[int(i+1):size]
        ri = r[int(i+1):size]
        W_BWi = W_BW[int(i+1):size]
        
        mu = np.dot(ri,cent)
        theta = np.arccos(mu)*180/np.pi
        
        new,indices = npi.group_by(W_BWi,np.arange(len(W_BWi)))
        N2 = np.array(list((map(PIP,new,itertools.repeat(W_BWp)))))
        wpip = np.hstack([np.repeat(x,len(y)) for x,y in zip(N2,indices)])
        thetax = np.hstack([theta[ind] for ind in indices])
        wix = np.hstack([wi[ind] for ind in indices])

        hist,_,_= binned_statistic(thetax,wix*wcent*wpip,statistic='sum',bins=binsfile)
        count_theta += hist
        normDD += np.sum(wix*wpip)*wcent
        
    return normDD,count_theta


def DD_brute_rppi_ANGPIP(*args,**qwargs):
    ra = args[0]
    dec = args[1]
    Dc = args[2]
    w =args[3]
    binsfile = args[4]
    pibin = args[5]
    W_BW = args[6]
    scaling = args[7]
    x = Dc*np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180)
    y = Dc*np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180)
    z = Dc*np.sin(dec*np.pi/180)
    r = np.column_stack([x,y,z])
    size = x.size
        
    count_rppi = np.zeros((len(binsfile)-1,len(pibin)-1))
    normDD = 0
    
    for i in range(0,size-1):
        cent,wcent,W_BWp = r[i],w[i],W_BW[i]
        wi = w[int(i+1):size]
        ri = r[int(i+1):size]
        W_BWi = W_BW[int(i+1):size]

        s = distance(cent,ri)
        los = (cent + ri)/2.
        ss = (cent - ri)
        los = los/norm(los)[:,None]
        ss = ss/norm(ss)[:,None]
        mu = abs(np.array([np.dot(x,y) for x,y in zip(los,ss)]))
        pi = s*mu
        rp = np.sqrt(s**2 - pi**2)
        
        new,indices = npi.group_by(W_BWi,np.arange(len(W_BWi)))
        N2 = np.array(list((map(PIP,new,itertools.repeat(W_BWp)))))
        wpip = np.hstack([np.repeat(x,len(y)) for x,y in zip(N2,indices)])
        rpx = np.hstack([rp[ind] for ind in indices])
        pix = np.hstack([pi[ind] for ind in indices])
        wix = np.hstack([wi[ind] for ind in indices])
        
        hist,_,_,_= binned_statistic_2d(rpx,pix,wix*wcent*wpip,statistic='sum',bins=[binsfile,pibin])
        count_rppi += scaling[:,None]*hist
        normDD += np.sum(wix*wcent*wpip)
        
    return normDD,count_rppi