import numpy as np
from scipy import stats
"""Ugly pipeline to create chuncks"""
"""Return the same catalog with chunck number as additional column"""

class jaccky:
    
    def __init__(self,*args,**qwargs):


        self.ra_jack = []
        self.dec_jack = []
        self.z_jack = []
        self.w_jack = []
        self.e1_jack =[]
        self.e2_jack = []
        self.z_jack = []
        self.number_jack = []

        self.ra = args[0]
        self.dec = args[1]
        self.z = args[2]
        self.w = args[3]
        self.e1 = args[4]
        self.e2 = args[5]
        self.npatch = qwargs['npatch']
        self.sample_size = int(len(self.ra)/10)
        self.decbins = np.linspace(self.dec.min(),self.dec.max(),int(self.npatch/2 + 1.))
        
        
    def get_samples(self):
        bin_dec = stats.binned_statistic(self.dec,self.dec, statistic="count", bins=self.decbins)
        data = np.column_stack([self.ra,self.dec,self.z,self.w,self.e1,self.e2,bin_dec[2].astype(int)])
        data = data[data[:, 6].argsort()]
        databinned = np.split(data, np.unique(data[:,6], return_index = True)[1])[1:]

        k = 0
        for i in range(0,len(databinned)):
            array = np.array(databinned[i])
            array = array[array[:, 0].argsort()]
            sub = int(len(array[:,0])/self.sample_size)
            new = np.array_split(array,sub)
            for j in range(0,len(new)):
                arrayj = np.array(new[j])
                patch = arrayj[:,6] + i + j 
                self.ra_jack.append(arrayj[:,0])
                self.dec_jack.append(arrayj[:,1])
                self.z_jack.append(arrayj[:,2])
                self.w_jack.append(arrayj[:,3])
                self.e1_jack.append(arrayj[:,4])
                self.e2_jack.append(arrayj[:,5])
                self.number_jack.append(patch)

        self.ra_jack = np.hstack(self.ra_jack)
        self.dec_jack = np.hstack(self.dec_jack)
        self.z_jack = np.hstack(self.z_jack)
        self.w_jack = np.hstack(self.z_jack)
        self.e1_jack = np.hstack(self.e1_jack)
        self.e2_jack = np.hstack(self.e2_jack)
        self.number_jack = np.hstack(self.number_jack)
        
        self.dataout = np.column_stack([self.ra_jack,self.dec_jack,self.z_jack,self.w_jack,self.e1_jack,self.e2_jack,self.number_jack])
        return self.dataout



    """Jack the clustering catalog if needed"""
        
        
        