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

T = Table.read(pathELG)


"""Clustering catalog"""
cond = np.where((T['CLUSTERING']==1) & (T['Z'] > 0.6) & (T['Z'] < 1.1))

Tc = T[cond]
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

cond = ((T2['CLUSTERING']==1) & (T2['Z'] > 0.6) & (T2['Z'] < 1.1))
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
    return Nbits/np.sum(Nn)

def PIP(x,y):
    count = (x & y)
    Nn = np.unpackbits(count.view('uint8'))
    return Nbits/np.sum(Nn)


wIIP= np.array(list(map(IIP,W_BW)))


def distance(cent,pos):
    dist = np.sqrt((pos[:,0] - cent[0])**2+(pos[:,1] - cent[1])**2+(pos[:,2] - cent[2])**2)
    return dist
    
def norm(vec):
    return np.sqrt(vec[:,0]**2+vec[:,1]**2+vec[:,2]**2)

def normr(vec):
    return np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)



def DD_brute_rppi(*args,**qwargs):
    ra = args[0]
    dec = args[1]
    Dc = args[2]
    w =args[3]
    binsfile = args[4]
    pibin = args[5]
    x = Dc*np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180)
    y = Dc*np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180)
    z = Dc*np.sin(dec*np.pi/180)
    r = np.column_stack([x,y,z])
    size = x.size
        
    count_rppi = np.zeros((len(binsfile)-1,len(pibin)-1))
    normDD = 0
    
    for i in range(0,size-1):
        cent,wcent = r[i],w[i]
        wi = w[int(i+1):size]
        ri = r[int(i+1):size]

        s = distance(cent,ri)
        los = (cent + ri)/2.
        ss = (cent - ri)
        los = los/norm(los)[:,None]
        ss = ss/norm(ss)[:,None]
        mu = abs(np.array([np.dot(x,y) for x,y in zip(los,ss)]))
        pi = s*mu
        rp = np.sqrt(s**2 - pi**2)
        
        hist,_,_,_= binned_statistic_2d(rp,pi,wi*wcent,statistic='sum',bins=[binsfile,pibin])
        count_rppi += hist
        normDD += np.sum(wi*wcent)
        
    return normDD,count_rppi


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


"""logbins for wp(rp)"""

theta,scalingDD,scalingDR = np.loadtxt('/feynman/work/dap/lceg/rp269101/stuff/angular_count_logbins_ELG.dat',unpack=True)


pi_max = 80
rpbins =  np.geomspace(0.2,50,13)
pibins = np.linspace(0,pimax,pimax+1)

dx = np.diff(rpbins);   rp = bins[0:-1] + 0.5*dxc

""" PIP wp(rp)"""

autocorr=0

normDD,DD = DD_brute_rppi_ANGPIP(ra,dec,Dc,w,rpbins,pibins,W_BW,scaling)


RRrp = DDrppi_mocks(autocorr,1,12,pi_max,rpbins,
                        ra_r,dec_r,Dc_r,weights1=w_r,
                               weight_type='pair_product',is_comoving_dist=True,
                         output_rpavg=True)



autocorr=1


DRrp = DDrppi_mocks(autocorr,1,12,pi_max,rpbins,ra,dec,Dc,weights1 = w*wIIP,\
                    RA2 = ra_r, DEC2=dec_r,\
                      CZ2=Dc_r,weight_type='pair_product',weights2=w_r,\
                      is_comoving_dist=True,output_rpavg=True)


DR = scalingDR[:,None]*DRrp['npairs'].reshape(len(rpbins),pi_max)
wDR = DRrp['weightavg'].reshape(len(rpbins),pi_max)

RR = RRrp['npairs'].reshape(len(rpbins),pi_max)
wRR = RRrp['weightavg'].reshape(len(rpbins),pi_max)



normRR = np.sum(w_r)**2 - np.sum(w_r**2)
normDR = np.sum(w_r)*np.sum(w*wIIP)



xirppi = DD/(RR*wRR)*(normRR/normDD) -2*(DR*wDR)/(RR*wRR)*(normRR/normDR) + 1
wp = np.sum(xirppi,axis = 1)

""" wp(rp) cp weights """

autocorr=0
RRrp = DDrppi_mocks(autocorr,1,12,pi_max,rpbins,
                        ra_r,dec_r,Dc_r,weights1=w_r*wcp_r,
                               weight_type='pair_product',is_comoving_dist=True,
                         output_rpavg=True)


DDrp = DDrppi_mocks(autocorr,1,12,pi_max,rpbins,
                        ra,dec,Dc,weights1=w*wcp,
                               weight_type='pair_product',is_comoving_dist=True,
                         output_rpavg=True)

autocorr=1


DRrp = DDrppi_mocks(autocorr,1,4,pi_max,rpbins,ra,dec,Dc, weights1 = w*wcp,\
                    RA2 = ra_r, DEC2=dec_r,\
                      CZ2=Dc_r,weight_type='pair_product',weights2=w_r*wcp_r,\
                      is_comoving_dist=True,output_rpavg=True)


DD = DDrp['npairs'].reshape(len(rpbins),pi_max)
wDD = DRrp['weightavg'].reshape(len(rpbins),pi_max)

DR = DRrp['npairs'].reshape(len(rpbins),pi_max)
wDR = DRrp['weightavg'].reshape(len(rpbins),pi_max)

RR = RRrp['npairs'].reshape(len(rpbins),pi_max)
wRR = RRrp['weightavg'].reshape(len(rpbins),pi_max)


normRR = np.sum(w_r*wcp_r)**2 - np.sum((w_r*wcp_r)**2)
normDD = np.sum(w*wcp)**2 - np.sum((w*wcp)**2)
normDR = np.sum(w_r*wcp_r)*np.sum(w*wcp)


xirppi = (DD*wDD)/(RR*wRR)*(normRR/normDD) -2*(DR*wDR)/(RR*wRR)*(normRR/normDR) + 1
wp2 = np.sum(xirppi,axis = 1)

np.savetxt('/feynman/work/dap/lceg/rp269101/stuff/wp_ELG_SGC.dat',np.transpose([rp,wp,wp2]))


"""linear bins for xi(s,mu)"""

pi_max = 150
rpbins =  np.linspace(1e-5,pimax+1e-5,601)
pibins = rpbins

dx = np.diff(rpbins);   rp = bins[0:-1] + 0.5*dxc
pi = rp

rpx,pix = np.meshgrid(rp,pi)

theta,scalingDD,scalingDR = 
np.loadtxt('/feynman/work/dap/lceg/rp269101/stuff/angular_count_linearbin_ELG.dat',unpack=True)


autocorr=0

normDD,DD = DD_brute_rppi_ANGPIP(ra,dec,Dc,w,rpbins,pibins,W_BW,scaling)


RRrp = DDrppi_mocks(autocorr,1,12,pi_max,rpbins,
                        ra_r,dec_r,Dc_r,weights1=w_r,
                               weight_type='pair_product',is_comoving_dist=True,
                         output_rpavg=True)



autocorr=1


DRrp = DDrppi_mocks(autocorr,1,12,pi_max,rpbins,ra,dec,Dc,weights1 = w*wIIP,\
                    RA2 = ra_r, DEC2=dec_r,\
                      CZ2=Dc_r,weight_type='pair_product',weights2=w_r,\
                      is_comoving_dist=True,output_rpavg=True)


DR = scalingDR[:,None]*DRrp['npairs'].reshape(len(rpbins),pi_max)
wDR = DRrp['weightavg'].reshape(len(rpbins),pi_max)

RR = RRrp['npairs'].reshape(len(rpbins),pi_max)
wRR = RRrp['weightavg'].reshape(len(rpbins),pi_max)



normRR = np.sum(w_r)**2 - np.sum(w_r**2)
normDR = np.sum(w_r)*np.sum(w*wIIP)



xirppi = DD/(RR*wRR)*(normRR/normDD) -2*(DR*wDR)/(RR*wRR)*(normRR/normDR) + 1

np.savetxt('/feynman/work/dap/lceg/rp269101/stuff/xirppi_ELG_PIP.dat',np.transpose([rpx.flatten(),pi.flatten(),xirppi.flatten()]))

