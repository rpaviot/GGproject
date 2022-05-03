import healpy as hp    
import numpy as np
from numpy.random import default_rng
rng = default_rng()
from numpy import rad2deg
from scipy.interpolate import CubicSpline as CS
from scipy import stats

"""Code to create random catalog for simple geometry."""

class survey_geometry:

    def __init__(self,*args,**qwargs):
        self.dc = None
        self.NSIDE = qwargs['NSIDE']
        self.ra = qwargs['RA']
        self.dec = qwargs['DEC']
        self.w = qwargs['W']
        if 'Z' in qwargs:
            self.z = qwargs['Z']
            self.zmin = np.around(min(self.z),decimals=2)
            self.zmax = np.around(max(self.z),decimals=2)

        else :
            self.dc = qwargs['Dc']
            self.dmin = min(self.dc)
            self.dmax = max(self.dc)

        self.target_nz = None
        self.maskdata = self.mask()
        
        self.ra_min = np.min(self.ra)
        self.ra_max = np.max(self.ra)
        self.dec_min = np.min(self.dec)
        self.dec_max = np.max(self.dec)
        

        self.redshift_CDF()


    """Target nz for random"""

    def set_target_nr(self,dc,weight):
        self.target_nz = 1
        npt = 51
        self.bins_data = np.linspace(self.dmin,self.dmax,npt)
        self.dn_data,_ = np.histogram(dc, bins=self.bins_data,weights=weight)
        self.redshift_CDF()


    """Cumalive distribution data and target must have same redshift range"""

    def redshift_CDF(self):
        npt = 51
        if self.target_nz is None and self.dc is None:
            dx = 0.005
            npt = int((self.zmax - self.zmin)/dx + 1.)
            bins = np.linspace(self.zmin,self.zmax,npt)
            dn, bins = np.histogram(self.z, bins=bins,weights=self.w)
            myPDF = dn/np.sum(dn)
            dxc = np.diff(bins);   xc = bins[0:-1] + 0.5*dxc
            myCDF = np.zeros(len(bins))
            myCDF[1:] = np.cumsum(myPDF)
            self.spline_inv = CS(myCDF,bins)           
        elif self.target_nz is None and self.dc is not None:
            bins = np.linspace(self.dmin,self.dmax,npt)
            dn, bins = np.histogram(self.dc, bins=bins,weights=self.w)
            myPDF = dn/np.sum(dn)
            dxc = np.diff(bins);   xc = bins[0:-1] + 0.5*dxc
            myCDF = np.zeros(len(bins))
            myCDF[1:] = np.cumsum(myPDF)
            self.spline_inv = CS(myCDF,bins)
        elif self.target_nz is not None:
            myPDF = self.dn_data/np.sum(self.dn_data)
            dxc = np.diff(self.bins_data);   xc = self.bins_data[0:-1] + 0.5*dxc
            myCDF = np.zeros(len(self.bins_data))
            myCDF[1:] = np.cumsum(myPDF)
            self.spline_inv = CS(myCDF,self.bins_data)

  
    """Subsample mock to get the same n(z) distribution"""
    def set_mock_nz(self,ra,dec,dc,w):
        print(len(ra))
        ra2 = []
        dec2 = []
        dc2 = []
        w2 = []
 
        dn_data2 = self.dn_data/10
        self.dn_mock,edges,binind= stats.binned_statistic(dc,w,bins=self.bins_data,statistic='sum')
        data = np.column_stack([ra,dec,dc,w,binind])
        data = data[data[:, 4].argsort()]
        databinned = np.split(data, np.unique(data[:,4], return_index = True)[1])[1:]

        div = self.dn_mock/dn_data2
        minh = np.min(div)


        for i in range(0,len(databinned)):
            databin = databinned[i]
            factor = minh/div[i]
            indices = np.arange(0,len(databin[:,0]))
            size = len(databin[:,0])
            sub = np.random.choice(indices,size=int(factor*size))
            databin = databin[sub]
            ra2.append(databin[:,0])
            dec2.append(databin[:,1])
            dc2.append(databin[:,2])
            w2.append(databin[:,3])
        ra2 = np.hstack(ra2)
        dec2 = np.hstack(dec2)
        dc2 = np.hstack(dc2)
        w2 = np.hstack(w2)
        print(len(ra2))
        return ra2,dec2,dc2,w2
              

    def DeclRaToIndex(self,ra,dec):
        return hp.pixelfunc.ang2pix(self.NSIDE,(-dec+90.)*np.pi/180.,ra*np.pi/180.)

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
        index = self.DeclRaToIndex(self.ra,self.dec)
        self.index = index
        footprint = np.unique(index)
        self.footprint = footprint
        p[footprint] =+1
        self.frac = np.sum(p)/len(p)
        print(self.frac)
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
 

    def create_random(self,size_random):

        size_random = int(size_random*1./self.frac)
        ra_rand = 360*rng.random(size_random)
        cosdec_rand = rng.random(size_random)
        dec_rand = rad2deg(np.arccos(cosdec_rand))
        ra_rand,dec_rand = self.infootprint(ra_rand,dec_rand)
        cond = np.where((ra_rand >= self.ra_min) & (ra_rand <= self.ra_max) & (dec_rand >= self.dec_min) & \
                        (dec_rand <= self.dec_max))
        ra_rand = ra_rand[cond]
        dec_rand = dec_rand[cond]
        
        random_numbers = rng.uniform(0,1,size=len(ra_rand))
        z_rand = self.spline_inv(random_numbers)
        w_rand = np.ones(len(z_rand))
        
        return ra_rand,dec_rand,z_rand,w_rand


    """Work only for simplest rectangular footprint without masks"""
    def create_random_easy(self,number):
        random_numbers = rng.uniform(0,1,size=int(number))
        self.z_rand = self.spline_inv(random_numbers)
        self.ra_rand = np.random.uniform(self.ra_min,self.ra_max,len(self.z_rand))
        self.sindec_rand = np.random.uniform(np.sin(self.dec_min*np.pi/180), np.sin(self.dec_max*np.pi/180),len(self.z_rand))
        self.dec_rand  = np.arcsin(self.sindec_rand)*180/np.pi
        self.w_rand = np.ones(len(self.ra_rand))
        return self.ra_rand,self.dec_rand,self.z_rand,self.w_rand

    
    def infootprint(self,ra,dec):
        index = self.DeclRaToIndex(ra,dec)
        cond = np.where(self.maskdata[index] !=0)
        ra2 = ra[cond]
        dec2 = dec[cond]
        return ra2,dec2

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
