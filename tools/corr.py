import numpy as np
import treecorr
from astropy.cosmology import FlatLambdaCDM


"""Code to compute gamma_t and wg+ for lightcones with treecorr"""
class lensingPCF:
    def __init__(self,*args,**qwargs):
        

        """Init : First catalog: clustering catalog, second catalog source catalog"""
        self.Om =  qwargs['Om0']
        self.cosmo = FlatLambdaCDM(Om0=self.Om,H0=100)
        
        self.computation = qwargs['computation']
        self.units = qwargs['units']
        self.npatch = qwargs['npatch']
        
        if self.npatch == 1:
            self.var_method = "shot"
        else:
            self.var_method = "jackknife"
 
        
        self.z = None
        self.z2 = None
        self.rand1 = None
        self.rand2 = None
        self.cov = None
        
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
            self.data1 = treecorr.Catalog(ra=ra,dec=dec,r=dc,w=w,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch)
        else :
            self.data1 = treecorr.Catalog(ra=ra,dec=dec,w=w,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch)

        
        if 'Z2' in qwargs:
            self.z2 =  qwargs['Z2']
            dc2 = self.cosmo.comoving_distance(self.z2).value
            self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,w=w2,g1=g1,g2=g2,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers)
        else :
            self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,w=w2,g1=g1,g2=g2,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers)
            

        self.varg = treecorr.calculateVarG(self.data2)
            

    def set_random(self,*args,**qwargs):
        ra_r =  qwargs['RA_r']
        dec_r =  qwargs['DEC_r']
        w_r =  qwargs['W_r']        
        if 'Z_r' in qwargs:
            z_r =  qwargs['Z_r']
            dc_r = self.cosmo.comoving_distance(z_r).value
            self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r,r=dc_r,w=w_r,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers,is_rand=1)
        else :
            self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r, w=w_r,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers,is_rand=1)

        if 'RA_r2' in qwargs:
            ra_r2 =  qwargs['RA_r2']
            dec_r2 =  qwargs['DEC_r2']
            w_r2 =  qwargs['W_r2']        
            if 'Z_r2' in qwargs:
                z_r2 =  qwargs['Z_r2']
                dc_r2 = self.cosmo.comoving_distance(z_r2).value
                self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,r=dc_r2,w=w_r2,ra_units=self.units,\
                    dec_units=self.units,npatch=1,patch_centers=self.data1.patch_centers,is_rand=1)
            else :
                self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,w=w_r2,ra_units=self.units,\
                    dec_units=self.units,npatch=1,patch_centers=self.data1.patch_centers,is_rand=1)


    def compute_norm(self):
        
        if self.rand1 is not None:
            self.rgnorm = self.rand1.sumw*self.data2.sumw
        
        if self.corr == "auto":
            self.ngnorm = (self.data1.sumw)**2 - np.sum((self.data1.w)**2)
            if self.rand1 is not None:
                self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
            else:
                raise ValueError("You must provide at least a random catalog")
        
        elif self.corr == "cross":
            if self.computation=="GG":
                self.ngnorm = self.data1.sumw*self.data2.sumw
                if self.rand1 is not None:
                    self.rrnorm = self.rand1.sumw*self.rand1.sumw 
                else:
                    raise ValueError("You must provide at least a random catalog")
                
            elif self.computation=="IA":
                if self.rand1 is not None and self.rand2 is not None :
                    self.ngnorm = self.data1.sumw*self.data2.sumw
                    self.rrnorm = self.rand1.sumw*self.rand2.sumw
                else :
                    raise ValueError("You must provide at least two random catalogs for wg+ cross estimation.")


    def compute_norm(self):
        
        if self.rand1 is not None:
            self.rgnorm = self.rand1.sumw*self.data2.sumw
        
        if self.corr == "auto":
            self.ngnorm = (self.data1.sumw)**2 - np.sum((self.data1.w)**2)
            if self.rand1 is not None:
                self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
            else:
                raise ValueError("You must provide at least a random catalog")
        
        elif self.corr == "cross":
            if self.computation=="GG":
                self.ngnorm = self.data1.sumw*self.data2.sumw
                if self.rand1 is not None:
                    self.rrnorm = self.rand1.sumw*self.rand1.sumw 
                else:
                    raise ValueError("You must provide at least a random catalog")
                
            elif self.computation=="IA":
                if self.rand1 is not None and self.rand2 is not None :
                    self.ngnorm = self.data1.sumw*self.data2.sumw
                    self.rrnorm = self.rand1.sumw*self.rand2.sumw
                else :
                    raise ValueError("You must provide at least two random catalogs for wg+ cross estimation.")

    def combine_pairs_DS(self,corrs):
        return corrs[0].xi*(corrs[0].weight/corrs[1].weight)*(self.rgnorm/self.ngnorm) - corrs[1].xi

    def combine_pairs_RS(self,corrs):
        return corrs[0].xi*(corrs[0].weight/corrs[2].weight)*(self.rrnorm/self.ngnorm) - \
            corrs[1].xi*(corrs[1].weight/corrs[2].weight)*(self.rrnorm/self.rgnorm)

    def combine_pairs_RS_proj(self,corrs):
        xirppi_t = np.zeros((len(self.pi)-1,self.nbins))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        rr = corrs[2*int(len(corrs)/3):len(corrs)]
        for i in range(0,len(self.pi)-1):
            corrs = [ng[i],rg[i],rr[i]]
            xirppi_t[i] = self.combine_pairs_DS(corrs)
        xirppi_t = np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t

    def gammat(self,minr,maxr,nbins,sep_units=None,min_rpar=0):
        min_rpar = min_rpar
        sep_units = sep_units
        self.compute_norm()
        """Distances sources > Distances lens + x Mpc """
     
        if self.z is None and self.rand1 is not None:
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,\
                sep_units=sep_units,var_method=self.var_method)

            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,\
                sep_units=sep_units,var_method=self.var_method)

            
            ng.process(self.data1,self.data2)
            rg.process(self.rand1,self.data2)
            
            
            corrs=[ng,rg]
            xi = self.combine_pairs_DS(corrs)
            if self.var_method =="jackknife":
                self.cov = treecorr.estimate_multi_cov(corrs, self.var_method, self.combine_pairs_DS)
                err = np.sqrt(np.diag(self.cov))
            else :
                err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))   
                
            rnorm = ng.rnom
            meanr = rg.meanr
            meanlogr =  rg.meanlogr
            
            
        elif self.z is not None and self.rand1 is not None:
            print("lol")
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp",var_method=self.var_method)
            
            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp",var_method=self.var_method)
            
            ng.process(self.data1,self.data2)
            rg.process(self.rand1,self.data2)
            
            corrs=[ng,rg]
            xi = self.combine_pairs_DS(corrs)
            if self.var_method =="jackknife":
                self.cov = treecorr.estimate_multi_cov(corrs, self.var_method, self.combine_pairs_DS)
                err = np.sqrt(np.diag(self.cov))
            else :
                err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))                
            rnorm = ng.rnom
            meanr = rg.meanr
            meanlogr =  rg.meanlogr

        return rg.rnom,meanr,meanlogr,xi,err

    def wgp(self,min_sep,max_sep,nbins,pimax,dpi):
        
        self.compute_norm()
        npt = int(2.*pimax/dpi) + 1
        """Treecorr provides 2D counts computation but only for linear bins here we compute xi(rp,pi) by looping over radial distance instead"""
        pi = np.linspace(-pimax,pimax,npt)
        self.pi = pi
        self.nbins = nbins
        self.dpi = pi[1] - pi[0]

        dictNG = {}
        dictRG = {}
        dictRR = {}
        
        for i in range(0,len(pi)-1):
            pi_min = pi[i]
            pi_max = pi[i+1]
            dictNG[i] = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)
            dictRG[i] = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)
            dictRR[i] = treecorr.NNCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)

            dictNG[i].process(self.data1,self.data2)
            dictRG[i].process(self.rand1,self.data2)
            if self.rand2 is None:
                dictRR[i].process(self.rand1,self.rand1)
            else :
                dictRR[i].process(self.rand1,self.rand2)
        catNG = list(dictNG.values())
        catRG =  list(dictRG.values())
        catRR = list(dictRR.values())
        corrs = catNG + catRG + catRR
        xi = self.combine_pairs_RS_proj(corrs)
        
        

        rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
        rg.process(self.rand1,self.data2)
        meanr = rg.meanr
        meanlogr =  rg.meanlogr
        
        if self.var_method == "jackknife":
            self.cov = treecorr.estimate_multi_cov(corrs, self.var_method, self.combine_pairs_RS_proj)
            err = np.sqrt(np.diag(self.cov))
        else :
            err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))   
            
        return dictNG[0].rnom,meanr,meanlogr,xi,err

    def get_cov(self):
        return self.cov


""" 
wrong implementation

class lensing2PCF_jack():


    def __init__(self,*args,**qwargs):
        self.Om =  qwargs['Om0']
        self.cosmo = FlatLambdaCDM(Om0=self.Om,H0=100)
        ra =  qwargs['RA']
        dec =  qwargs['DEC']
        w =  qwargs['W']
        npatch = qwargs['npatch']
        self.uniquepatch = np.unique(npatch)
        
        ra2 =  qwargs['RA2']
        dec2 =  qwargs['DEC2']
        w2 =  qwargs['W2'] 
        g1 = qwargs['g1']
        g2 = qwargs['g2']
        npatch2 = qwargs['npatch2']
        self.uniquepatch2 = np.unique(npatch2)

        if np.array_equal(ra,ra2):
            self.corr = "auto"
        else :
            self.corr = "cross"

        if 'Z' in qwargs:
            self.z =  qwargs['Z']
            dc = self.cosmo.comoving_distance(self.z).value
        
        if 'Z2' in qwargs:
            self.z2 =  qwargs['Z2']
            dc2 = self.cosmo.comoving_distance(self.z2).value

        self.computation = qwargs['computation']
        self.units = qwargs['units']
        self.minr = qwargs['minr']
        self.maxr = qwargs['maxr']
        self.nbins =qwargs['nbins']

        #define global matrice to store individual result

        if self.computation=="IA":
            if 'pimax' not in qwargs:
                ValueError("You must provide pimax and the number of pibins")
            else : 
                self.pimax = qwargs['pimax']
                self.npi = qwargs['npi']
                self.matT_DS = np.zeros((self.npatch,self.nbins,self.npi))
                self.matT_RS = np.zeros((self.npatch,self.nbins,self.npi))
                self.matT_DSw = np.zeros((self.npatch,self.nbins,self.npi))
                self.matT_RSw = np.zeros((self.npatch,self.nbins,self.npi))

        else:
            self.matT_DS = np.zeros((self.npatch,self.nbins))
            self.matT_RS = np.zeros((self.npatch,self.nbins))
            self.matT_DSw = np.zeros((self.npatch,self.nbins))
            self.matT_RSw = np.zeros((self.npatch,self.nbins))

        for i in range(0,self.uniquepatch):
            cond = np.where(self.uniquepatch == i+1)
            cond2 = np.where(self.uniquepatch2 == i+1)
            rax = ra[cond]
            decx = dec[cond]
            wx = w[cond]
            ray = ra2[cond2]
            decy = dec2[cond2]
            wy = w2[cond2]
            g1y = g1[cond2]
            g2y = g2[cond2]
            
            if self.z is not None:
                dcx = dc[cond]
                dcy = dc2[cond2]
                self.data1 = treecorr.Catalog(ra=ra,dec=dec,r=dc,w=w,ra_units=self.units,dec_units=self.units)
                self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,\
                                  w=w2,g1=g1,g2=g2,ra_units=self.units,dec_units=self.units)

            if self.z is None:
                self.data1 = treecorr.Catalog(ra=ra,dec=dec,w=w,ra_units=self.units,dec_units=self.units)
                self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,\
                                  w=w2,g1=g1,g2=g2,ra_units=self.units,dec_units=self.units)

            self.compute_pairs()

            self.matT_DS[i]=self.DSpairs
            self.matT_DSw[i]=self.DSweight
            self.matT_RS[i]=self.RSpairs
            self.matT_RSw[i]=self.RSweight



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


    def compute_pairs(self,sep_units=None,min_rpar=0):
        sep_units = sep_units
        min_rpar = min_rpar
        if self.computation == "GG":
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins,min_sep=self.minr,
                                          max_sep=self.maxr,min_rpar=min_rpar,metric="Rperp") 

            rg = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins,min_sep=self.minr,
                                          max_sep=self.maxr,min_rpar=min_rpar,metric="Rperp") 

            ng.process_cross(self.data1,self.data2)
            rg.process_cross(self.rand1,self.data2)

            self.DSpairs = ng.xi 
            self.DSweight = ng.weight

            self.RSpairs = rg.xi 
            self.RSweight = rg.weight
            

        else :
            pi = np.linspace(-self.pimax,self.pimax,self.npi)
            dpi = pi[1] - pi[0]
        
            self.DSpairs = np.zeros((len(pi),len(self.nbins)))
            self.DSweight = np.zeros((len(pi),len(self.nbins)))

            self.RSpairs = np.zeros((len(pi),len(self.nbins)))
            self.RSweight = np.zeros((len(pi),len(self.nbins)))

            for i in range(0,len(pi)-1):
                pi_min = pi[i]
                pi_max = pi[i] + dpi
                ng = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins,min_sep=self.min_sep,max_sep=self.max_sep,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
                rg = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins,min_sep=self.min_sep,max_sep=self.max_sep,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
                ng.process_cross(self.data1,self.data2)
                rg.process_cross(self.rand1,self.data2)
                self.DSpairs[i] = ng.xi
                self.DSweight[i] = ng.weight
                self.RSpairs[i] = rg.xi
                self.RSweight[i] = rg.weight
 """





   # def precompute_rr(self):

   # def combine(self):
