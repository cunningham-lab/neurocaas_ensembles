## Estimate the influence of individual datapoints on the ensemble prediction. One thing that's still not clear is how we go about quantifying the influence. 
## 1. replace the deviation with the evaluation of the probability at the ground truth marker point. 
## 2. replace the mean deviation with the ensemble deviation. 
## 3. repeat this process with the evaluation of the probability at the ground truth marker point according to the ensemble with and without a certain data point.  
import click
import os 
import json
from tqdm import tqdm
import numpy as np
import joblib
import matplotlib.pyplot as plt
from dgp_ensembletools.models import Ensemble
from pose_aum.datavis import DataInclusion
from compare_models_groundtruth import get_training_frames,parse_modelname

scriptdir = os.path.abspath(os.path.dirname(__file__))

modelpaths = "ensemble-model{i}-2030-01-0{i}"

@click.command("make matrix of data describing which models contain which datapoints.") 
@click.option("--video-name",default = "ibl1_labeled.mp4")
@click.option("--probabilities",default = os.path.join(scriptdir,"script_outputs","probability_data"))
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--ensemblesfolder",default = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21")
def main(video_name,probabilities,labellist,ensemblesfolder):
    """Runs after calculate_consensus script. Takes the performance estimates and ensembles from that data, and determines properties of the data from them.  

    """
    idstring = os.path.splitext(video_name)[0]+os.path.basename(os.path.normpath(ensemblesfolder))
    ensembles = joblib.load(probabilities)
    print(ensembles)

    ## Step 1: Next, figure out what ensembles see which training data:  
    all_frames = []
    per_ensemble = {}
    for e in ensembles:
        nb_frames = e["frames"]
        seeds = e["seed"]
        template = e["template"]## per- run template. because we prefix some runs as dgpreal2
        train_frames = get_training_frames(nb_frames,[seeds],ensemblesfolder,ensemblesfolder,video_name.split("_labeled.mp4")[0])
        ## Quick check: assert that these frames are in the appropriate model folder: 
        for t in train_frames:
            datafolder = os.path.join(ensemblesfolder,template.format(f =nb_frames, s = seeds),"process_results",modelpaths.format(i=1),"labeled-data",video_name.split("_labeled.mp4")[0])
            contents = os.listdir(datafolder)
            frame_id = "img{0:03d}.png".format(t)
            assert frame_id in contents, "Mismatch between extracted training data and data found in labeled data folder"
        all_frames.extend(train_frames)
        per_ensemble["seed: {}, frames: {}".format(seeds,nb_frames)] = {"frame_inds":train_frames,"performance":0} ## we won't use performance variable for now. 

    di = DataInclusion(per_ensemble)
    mat,inds,seeds = di.make_mat()
    ensembleparams,frames = mat.shape
    frame_mapping = {}
    for f in range(frames): 
        including = mat[:,f]
        excluding = ~mat[:,f]
        params_including = np.array(seeds)[including]
        params_excluding = np.array(seeds)[excluding]
        frame_mapping[f] = {"include":params_including,"exclude":params_excluding}

    ## Step 2: Now calculate mean deviations for including and excluding:
    splits = {}
    delta_probabilities = {}
    frameorder = []
    for frame,fdict in tqdm(frame_mapping.items()):
        include = []
        exclude = []
        for e in ensembles:
            seedframestring = "seed: {}, frames: {}".format(e["seed"],e["frames"])
            if seedframestring in fdict["include"]:    
                include.extend(e["modeldiffs"].values()) ## modeldiffs is an array of shape 1000,4
            elif seedframestring in fdict["exclude"]: 
                exclude.extend(e["modeldiffs"].values())
            else:
                raise Exception("error in parsing paths! ")
        ## We should take the norm, then calculate statistics..     
        mean_include = np.mean(np.array(include),axis = 0)
        mean_exclude = np.mean(np.array(exclude),axis = 0)
        delta_probability = mean_include-mean_exclude

        delta_probabilities[frame] = delta_probability
        ## Finally, save the raw partition: 
        #splits[frame] = {"include":[gt-i for i in include],"exclude":[gt-e for e in exclude]} ## must be gt - difference bc we calculated as gt - detection before. 
    probarrayrep = np.array([v for v in delta_probabilities.values()]).squeeze()  ## (of shape nb_frames,1000,4)  
    #normed = np.linalg.norm(arrayrep,axis = 2) ## now as distances in xy

    fig,ax = plt.subplots(probarrayrep.shape[-1],1,figsize = (25,25))
    for i in range(probarrayrep.shape[-1]):
        mappable = ax[i].matshow(probarrayrep[:,:,i],aspect = "auto")
        ax[i].set_yticks(range(len(frame_mapping)))
        ax[i].set_yticklabels(inds)
        fig.colorbar(mappable,ax=ax[i])
    plt.title("Marginal change in probability as a function of training frame inclusion")    
    plt.tight_layout()    
    plt.savefig(os.path.join(scriptdir,"../","images/","influence_probability_mat_{}".format(idstring)))
    plt.close()

    ## Additionally, plot per-frame histograms of the influence based probability:
    for i in range(len(inds)):
        try:
            plt.hist(probarrayrep[i,:,:].flatten())
            plt.title("Histogam of probability influences for Frame {}".format(inds[i]))
            plt.savefig(os.path.join(scriptdir,"../","images/","influence_probability_frame{i}_hist_{s}.png".format(i = inds[i],s=idstring)))
            plt.close()
        except ValueError: # most likely this is one of the frames that have nans in their exclude column:     
            continue


    joblib.dump({"frame_index":inds,"delta_prob":probarrayrep},os.path.join(scriptdir,"script_outputs","delta_probability_data_{}".format(idstring)))

if __name__ == "__main__":
    main()
