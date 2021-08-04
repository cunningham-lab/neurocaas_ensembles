# Script to calculate power and cost. 
import click
import pickle
import imageio
from scipy.stats import hypergeom
from statsmodels.stats.power import TTestIndPower
import os
import numpy as np
import matplotlib.pyplot as plt

filepath = os.path.abspath(os.path.dirname(__file__))
repopath = os.path.dirname(filepath)

"""Usage

:param labellist: a pickled list of the frame id, and correspondence to three different outlier criteria. 
:param dist1: a csv file containing the approximate rmses for distribution without outlier 
:param dist2: a csv file containing the approximate rmses for distribution with outlier

"""

@click.command(help = "Pre-analysis evaluation of pilot data")
@click.option("--labellist",
        type=click.Path(file_okay = True,dir_okay = False,readable = True),
        help = "a pickled list of the frame id, and correspondence to three different outlier criteria.",
        default = os.path.join(repopath,"data/ibl/classified_list")
        )
@click.option("--dist1",
        type=click.Path(file_okay = True,dir_okay = False,readable = True),
        help = "a csv file containing the approximate rmses for distribution 1",
        default = os.path.join(repopath,"data/ibl/rmse_10")
        )
@click.option("--dist2",
        type=click.Path(file_okay = True,dir_okay = False,readable = True),
        help = "a csv file containing the approximate rmses for distribution 2",
        default = os.path.join(repopath,"data/ibl/rmse_30")
        )
def main(labellist,dist1,dist2):
    ## First get the list:  
    with open(labellist,"rb") as fp:
        labels = pickle.load(fp)
    ## Calculate the proportion of outliers in the data to begin with     
    proportion = calculate_proportion_outliers(labels)    
    ## Power analysis on this data 
    stripnew = lambda x: int(x.split("\n")[0])
    with open(dist1,"r") as fp:
        dist1vals = list(map(stripnew,fp.readlines()))
    with open(dist2,"r") as fp:
        dist2vals = list(map(stripnew,fp.readlines()))
    n_total = power_analysis(proportion,dist1vals,dist2vals)    
    ## Finally calculate cost assuming 6 hours per network 
    cost = n_total*6*3.5
    print("Estimated cost: ${}".format(cost))
    
        

def calculate_proportion_outliers(labels):
    """Given some list of lists containing label ids, and outlier indicators [["1.png",0,0,1],["2.png",1,0,1]], calculates the expected proportion of subsamples of 10% and 30% minibatches that should contain at least one outlier. Uses the hypergeometric distribution to calculate the distributions involved, and averages them. It does not matter that the 10% samples are a subset of the 30% samples when calculating this proportion. 
 
    :param labels: list of data. 
    :returns: float: the proportion of training data subsamples we expect to contain an outlier.
    """
    count_outliers = 0
    for item in labels:
        outliers = item[1:]
        outfloat = map(float,outliers)
        if sum(outfloat) > 0:
            count_outliers += 1
    r = count_outliers/len(labels)
    M = len(labels)
    print("\n1. {} portion of labeled data are outliers.".format(r))        
    
    params_10 = [M,int(r*M),int(0.1*M)] #hypergeometric parameters for the pdf of the probability that there are no outliers in the 10% distribution 
    params_30 = [M,int(r*M),int(0.3*M)] #hypergeometric parameters for the pdf of the probability that there are no outliers in the 30% distribution (subsetting does not matter)

    rv_10 = hypergeom(*params_10)
    rv_30 = hypergeom(*params_30)

    p_10 = 1-rv_10.pmf(0)
    p_30 = 1- rv_30.pmf(0)
    proportion_minibatch = 0.5*p_10+0.5*p_30
    print("\n2. Proportion of batches that contain an outlier: {}".format(proportion_minibatch))
    return proportion_minibatch

def power_analysis(proportion_minibatch,dist1vals,dist2vals):
    """Given the proportion of samples we expect to contain an outlier and estimated distribution of rmses for both outlier and non-outlier populations, performs a power analysis at p = 0.05

    :param proportion_minibatch: the proportion of samples that are expected to contain an outlier. 
    :param dist1vals: a list of values for the distribution without outliers. 
    :param dist2vals: a list of values for the distribution with outliers. 
    :returns: return the total number of networks we need to run. 

    """
    ## First, calculate the effect size from samples as Glass' delta (only std from control group): 
    delta = abs((np.mean(dist2vals)-np.mean(dist1vals))/np.std(dist1vals))
    print("\n3. Effect size (Glass's delta): {}".format(delta))
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size = delta,nobs1 = None,alpha = 0.05,power = 0.8,ratio = (1-proportion_minibatch)/proportion_minibatch)
    print("\nCONCLUSION: We require {} samples of non-outliers, and {} samples of outliers for a significance level of 0.05, assuming power of 0.8 and that outlier existence is the driving force in performance variance across different networks.\n".format(np.ceil(n),np.ceil(proportion_minibatch*n)))
    return np.ceil(n)+np.ceil(proportion_minibatch*n)
    


if __name__ == "__main__":
    main()


