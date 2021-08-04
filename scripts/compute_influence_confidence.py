## Evaluate groundtruth marker point confidence on a GPU machine. 
import click
import os 
import json
from tqdm import tqdm
import numpy as np
import joblib
import matplotlib.pyplot as plt
from dgp_ensembletools.models import Ensemble
from dgp_ensembletools.datavis import DataInclusion
from compare_models_groundtruth import get_training_frames,parse_modelname

scriptdir = os.path.abspath(os.path.dirname(__file__))

modelpaths = "ensemble-model{i}-2030-01-0{i}"

@click.command("make a dictionary containing the confidence estimates at the groundtruth location") 
@click.option("--video-name",default = "ibl1_labeled.mp4")
@click.option("--groundtruth",default = "../data/ibl/ibl1_true_xy_all_918pm.mat")
@click.option("--partperm",default = "ibl",help="if we need to permute the labels of the groundtruth before comparing.")
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--ensemblesfolder",default = "/home/ubuntu/july_data")
def main(video_name,groundtruth,partperm,labellist,ensemblesfolder):
    """Runs after calculate_consensus script. Takes the performance estimates and ensembles from that data, and determines properties of the data from them.  

    """
    idstring = os.path.splitext(video_name)[0]+os.path.basename(os.path.normpath(ensemblesfolder))

    if partperm == "ibl":
        partperm = np.array([1,3,0,2]) # permute parts before comparing
    else:
        partperm = None

    ## Step 0: First get all diffs:     
    ensembles = [parse_modelname(m,labellist,ensemblesfolder) for m in os.listdir(ensemblesfolder) if parse_modelname(m,labellist,ensemblesfolder) is not None]
    gt = ensembles[0]["models"][0].get_groundtruth(groundtruth,partperm =partperm )

    for e in ensembles:
        e["modeldiffs"] = {"model1":None,"model2":None}
        for mi,m in enumerate(e["models"]):
            diff = m.get_groundtruth_confidence(groundtruth,video_name,range(1001),partperm = [1,3,0,2])
            e["modeldiffs"]["model{}".format(mi+1)] = diff
            
    joblib.dump(ensembles,os.path.join(scriptdir,"script_outputs","confidence_data_{}".format(idstring)))


if __name__ == "__main__":
    main()
