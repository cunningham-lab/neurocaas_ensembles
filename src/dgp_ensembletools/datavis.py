## data visualization module

import numpy as np
import matplotlib.pyplot as plt


class DataInclusion():
    """Class to visualize membership of different data frames in a subset of training data. 

    :param datadict: a dictionary where keys indicate ensemble seeds, and values are dictionaries with the keys [frame_inds,performance] with corresponding entries of type () 
    :ivar datadict: initial value: datadict
    :ivar all_frames: initial value: self.get_all_frames()

    """

    def __init__(self,datadict):
        self.datadict = datadict
        self.all_frames = self.get_all_frames()
    
    def get_all_frames(self):
        """Get the list of all training frame indices: 

        """
        all_frames = []
        for s,si in self.datadict.items():
            all_frames.extend(si["frame_inds"])
        return set(all_frames)    

    def make_mat(self):
        """Extract out the data and return a numpy array for it. Rows represent ensembles, and columns frame indices. 

        :returns (numpy array, label indices, ensemble seeds):
        """
        all_frames = self.get_all_frames()
        ## Create a boolean index of if the ensemble contains a given set of training frames: 
        mat = []
        seeds = []
        for seed,vals in self.datadict.items():
            inds = vals["frame_inds"]
            boollist = [afi in inds for afi in all_frames]
            mat.append(boollist)
            seeds.append(seed)
        return np.array(mat),list(all_frames),seeds    

    def make_sorted_mat(self):
        """Extract out the data and return a row-sorted (by performance) and column sorted (by frame index) numpy array for it. Rows represent ensembles in descending order of performance (higher is better), and columns frame indices. Assume performance is given as a cost. 

        """
        get_perf = lambda x: x[1]["performance"]
        all_frames = self.get_all_frames()
        ## Create a boolean index of if the ensemble contains a given set of training frames: 
        mat = []
        seeds = []
        ## Sort dictionary keys by performance: 
        sorted_items = sorted(self.datadict.items(),key = get_perf)
        for seed,vals in sorted_items:
            inds = vals["frame_inds"]
            boollist = np.array([afi in inds for afi in sorted(all_frames)])
            mat.append(boollist)
            seeds.append(seed)
        return np.array(mat),list(sorted(all_frames)),seeds    
        

                    



