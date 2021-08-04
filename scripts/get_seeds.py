## Script to get the seeds we will use to analyze minibatches. 
"""This script takes as input a list of labeled data with outliers indicated, and sequence of seeds to use for random generation  

"""
import os
import numpy as np
import pickle
import numpy 
import click 

filepath = os.path.abspath(os.path.dirname(__file__))
repopath = os.path.dirname(filepath)

@click.command(help = "Calculate outlier inclusion proportion for given seeds.")
@click.option("--seed","-s",multiple=True)
@click.option("--labellist",
        type=click.Path(file_okay = True,dir_okay = False,readable = True),
        help = "a pickled list of the frame id, and correspondence to three different outlier criteria.",
        default = os.path.join(repopath,"data/ibl/classified_list")
        )
def main(seed,labellist):
    ## First get the list:  
    with open(labellist,"rb") as fp:
        labels = pickle.load(fp)
    print(len(labels), "labels in full training dataset.")    
    nb_datapoints = len(labels)
    ## TODO here get a simple indicator of if the element is an outlier or not.     
    is_outlier = lambda x: sum(map(float,x[1:]))>0
    outliers = np.array(list(map(is_outlier,labels))) ## boolean list of if outlier or not. 

    seeddict = {}
    outliers_count = 0
    no_outliers_count = 0 
    tally = [no_outliers_count,outliers_count]
    p10 = int(nb_datapoints*0.1)
    p30 = int(nb_datapoints*0.3)
    print("10% split: {} frames. 30% split: {} frames".format(p10,p30))
    
    for s in seed:    
        si = int(s)
        np.random.seed(si)
        sequence10 = np.sort(np.random.choice(nb_datapoints,p10,replace=False))
        np.random.seed(si)
        sequence30 = np.sort(np.random.choice(nb_datapoints,p30,replace=False))
        print(np.array(labels)[sequence30])
        np.random.seed()
        ## determine if they contain outliers. 
        outliers_10 = np.any(outliers[sequence10]) ## True or False
        outliers_30 = np.any(outliers[sequence30]) ## True or False
        tally[outliers_10]+=1
        tally[outliers_30]+=1
        seeddict[si] = {10:outliers_10,30:outliers_30}
    print("Batches for each seed contain outliers: {}".format(seeddict))
    print("Total number of samples without/with outliers: {}".format(tally))

if __name__ == "__main__":
    main()

