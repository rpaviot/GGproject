import numpy as np
import treecorr
from astropy.cosmology import FlatLambdaCDM


"""Code to compute gamma_t and wg+ for light cones. Shot noice only for now"""

class lensingPCF:
    def __init__(self,*args,**qwargs):

        """Init : First catalog: clustering catalog, second catalog source catalog"""
        self.Om =  qwargs['Om0']
        self.cosmo = FlatLambdaCDM(Om0=self.Om,H0=100)
        self.z = None
        self.z2 = None
        self.rand1 = None
        self.computation = qwargs['computation']
        self.units = qwargs['units']

        ra =  qwargs['RA']
        dec =  qwargs['DEC']
        w =  qwargs['W']
        len1 = len(ra)
        
        ra2 =  qwargs['RA2']
        dec2 =  qwargs['DEC2']
        w2 =  qwargs['W2'] 
        g1 = qwargs['g1']
        g2 = qwargs['g2']
        len2 = len(ra2)
        
        if np.array_equal(ra,ra2):
            self.corr = "auto"
        else :
            self.corr = "cross"
    
        if 'Z' in qwargs:
            self.z =  qwargs['Z']
            dc = self.cosmo.comoving_distance(self.z).value
            self.data1 = treecorr.Catalog(ra=ra,dec=dec,r=dc,w=w,ra_units=self.units,dec_units=self.units)
        else :
            self.data1 = treecorr.Catalog(ra=ra,dec=dec,w=w,ra_units=self.units,dec_units=self.units)

            
        if 'Z2' in qwargs:
            self.z2 =  qwargs['Z2']
            dc2 = self.cosmo.comoving_distance(self.z2).value
            self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,\
                                  w=w2,g1=g1,g2=g2,ra_units=self.units,dec_units=self.units)
        else :
            self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,\
                                          w=w2,g1=g1,g2=g2,ra_units=self.units,dec_units=self.units)
            

        self.varg = treecorr.calculateVarG(self.data2)
            

    def set_random(self,*args,**qwargs):
        ra_r =  qwargs['RA_r']
        dec_r =  qwargs['DEC_r']
        w_r =  qwargs['W_r']        
        if 'Z_r' in qwargs:
            z_r =  qwargs['Z_r']
            dc_r = self.cosmo.comoving_distance(z_r).value
            self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r,r=dc_r,\
                                          w=w_r,ra_units=self.units,dec_units=self.units,is_rand=1)
        else :
            self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r,\
                                          w=w_r,ra_units=self.units,dec_units=self.units,is_rand=1)
        if 'RA_r2' in qwargs:
            ra_r2 =  qwargs['RA_r2']
            dec_r2 =  qwargs['DEC_r2']
            w_r2 =  qwargs['W_r2']        
            if 'Z_r2' in qwargs:
                z_r2 =  qwargs['Z_r2']
                dc_r2 = self.cosmo.comoving_distance(z_r2).value
                self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,r=dc_r2,w=w_r2,\
                                              ra_units=self.units,dec_units=self.units,is_rand=1)
            else :
                self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,w=w_r2,\
                                              ra_units=self.units,dec_units=self.units,is_rand=1)


    def compute_norm(self):
        
        if self.rand1 is not None:
            self.rgnorm = self.rand1.sumw*self.data2.sumw
        
        if self.corr == "auto":
            self.ngnorm = (self.data1.sumw)**2 - np.sum((self.data1.w)**2)
            if self.rand1 is not None:
                self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
        
        elif self.corr == "cross":
            if self.computation=="GG":
                self.ngnorm = self.data1.sumw*self.data2.sumw
                if self.rand1 is not None:
                    self.rrnorm = self.rand1.sumw*self.rand1.sumw     
                
            elif self.computation=="IA":
                if self.rand1 is None:
                    ValueError("You must provide at least a random catalog for wg+ estimation.")
                if self.rand2 == None:
                    self.ngnorm = self.data1.sumw*self.data2.sumw
                    self.rrnorm = self.rand1.sumw*self.rand1.sumw
                else :
                    self.ngnorm = self.data1.sumw*self.data2.sumw
                    self.rrnorm = self.rand1.sumw*self.rand2.sumw
                    
            



    def gammat(self,minr,maxr,nbins,sep_units=None,min_rpar=0):
        min_rpar = min_rpar
        sep_units = sep_units
        self.compute_norm()
        """Distances sources > Distances lens + 1 Mpc """
        if self.z is None and self.rand1 is None :
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,sep_units=sep_units)
            ng.process(self.data1,self.data2) 
            xi = ng.xi
            xix = ng.xi_im
            sigma = np.sqrt(ng.varxi)
            rnorm = ng.rnom
        
        elif self.z is None and self.rand1 is not None:
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,sep_units=sep_units)

            
            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,sep_units=sep_units)

            
            ng.process_cross(self.data1,self.data2)
            rg.process_cross(self.rand1,self.data2)
            
            xi = (ng.xi/rg.weight)*(self.rgnorm/self.ngnorm) - rg.xi/rg.weight
            xix = (ng.xi_im/rg.weight)*(self.rgnorm/self.ngnorm) - rg.xi_im/rg.weight
            sigma = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))
            rnorm = ng.rnom
            
        elif self.z is not None and self.rand1 is None:
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp") 
            ng.process(self.data1,self.data2) 
            xi = ng.xi
            xix = ng.xi_im
            sigma = np.sqrt(ng.varxi)
            rnorm = ng.rnom            

            
        elif self.z is not None and self.rand1 is not None:
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp") 
            
            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp") 
            
            ng.process_cross(self.data1,self.data2)
            rg.process_cross(self.rand1,self.data2)
            
            xi = (ng.xi/rg.weight)*(self.rgnorm/self.ngnorm) - rg.xi/rg.weight
            xix = (ng.xi_im/rg.weight)*(self.rgnorm/self.ngnorm) - rg.xi_im/rg.weight
            sigma = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))
            rnorm = ng.rnom
            


        return rnorm,xi,xix,sigma

    def wgp(self,min,max,nbins,pimax):
        
        self.compute_norm()
        """Treecorr provides 2D counts computation but only for linear bins here we compute xi(rp,pi) by looping over radial distance instead"""
        """10 Mpc pi bins like in the litterature"""
        pi = np.linspace(1,pimax+1,int(pimax/10)+1)

        #pi = np.linspace(-pimax,pimax,2*int(pimax/10)+1)
        dpi = pi[1] - pi[0]

        xirppi_t = np.zeros((len(pi),nbins))
        xirppi_x = np.zeros((len(pi),nbins))
        varg= np.zeros((len(pi),nbins))
        for i in range(0,len(pi)-1):
            pi_min = pi[i]
            pi_max = pi[i] + dpi
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min,max_sep=max,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min,max_sep=max,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
            ng.process_cross(self.data1,self.data2)
            rg.process_cross(self.rand1,self.data2)
  
            xirppi_t[i] = (ng.xi/rg.weight)*(self.rgnorm/self.ngnorm) - rg.xi/rg.weight
            xirppi_x[i] = (ng.xi_im/rg.weight)*(self.rgnorm/self.ngnorm) - rg.xi_im/rg.weight
            varg[i] = self.varg/rg.weight*(self.rgnorm/self.ngnorm)

        xit = np.sum(xirppi_t*dpi,axis=0)
        xip = np.sum(xirppi_x*dpi,axis=0)
        varg = np.sum(varg,axis=0)

        return rg.rnom,xit,xip,np.sqrt(varg)
