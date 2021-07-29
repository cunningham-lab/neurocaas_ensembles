## Estimate the influence of individual datapoints on the ensemble prediction. One thing that's still not clear is how we go about quantifying the influence. 

## First, try with distance of the provided detection to ground truth. 

## The algorithm goes something like this. 

## 0. For each ensemble, calculate the deviation from the ground truth as a time series. 

## 1. For each data point in the training set, partition all of the ensembles you have into those that include that data point, and those that do not. 

## 2. For each data point in the training set, calculate the mean deviation from the ground truth across 1_ the ensemble with and 2_ the ensemble without that data point.  

### Some next steps: 
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
from dgp_ensembletools.datavis import DataInclusion
from compare_models_groundtruth import get_training_frames,parse_modelname

scriptdir = os.path.abspath(os.path.dirname(__file__))

modelpaths = "ensemble-model{i}-2030-01-0{i}"

@click.command("make matrix of data describing which models contain which datapoints.") 
@click.option("--video-name",default = "ibl1_labeled.mp4")
@click.option("--groundtruth",default = "../data/ibl/ibl1_true_xy_all_918pm.mat")
@click.option("--partperm",default = "ibl",help="if we need to permute the labels of the groundtruth before comparing.")
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--basefolder",default ="/home/ubuntu/july_data")
@click.option("--resultsfolder",help="path to folder containing json files of outputs.",default = "../data/ibl/consensus_performance")
def main(video_name,groundtruth,partperm,labellist,basefolder,resultsfolder):
    """Runs after calculate_consensus script. Takes the performance estimates and ensembles from that data, and determines properties of the data from them.  

    """
    if partperm == "ibl":
        partperm = np.array([1,3,0,2]) # permute parts before comparing
    else:
        partperm = None

    ## Step 0: First get all diffs:     
    ensembles = [parse_modelname(m,labellist,basefolder) for m in os.listdir(basefolder) if parse_modelname(m,labellist,basefolder) is not None]
    gt = ensembles[0]["models"][0].get_groundtruth(groundtruth,partperm =partperm )

    for e in ensembles:
        e["modeldiffs"] = {"model1":None,"model2":None}
        for mi,m in enumerate(e["models"]):
            diff = m.get_groundtruth_confidence(groundtruth,video_name,range(1001),partperm = [1,3,0,2])
            e["modeldiffs"]["model{}".format(mi+1)] = diff
            
    joblib.dump(ensembles,os.path.join(scriptdir,"script_outputs","confidence_data"))

    ### Step 1: Next, figure out what ensembles see which training data:  
    #all_frames = []
    #per_ensemble = {}
    #for e in ensembles:
    #    nb_frames = e["frames"]
    #    seeds = e["seed"]
    #    template = e["template"]## per- run template. because we prefix some runs as dgpreal2
    #    train_frames = get_training_frames(nb_frames,[seeds],basefolder,basefolder,video_name.split("_labeled.mp4")[0])
    #    ## Quick check: assert that these frames are in the appropriate model folder: 
    #    for t in train_frames:
    #        datafolder = os.path.join(basefolder,template.format(f =nb_frames, s = seeds),"process_results",modelpaths.format(i=1),"labeled-data",video_name.split("_labeled.mp4")[0])
    #        contents = os.listdir(datafolder)
    #        frame_id = "img{0:03d}.png".format(t)
    #        assert frame_id in contents, "Mismatch between extracted training data and data found in labeled data folder"
    #    all_frames.extend(train_frames)
    #    per_ensemble["seed: {}, frames: {}".format(seeds,nb_frames)] = {"frame_inds":train_frames,"performance":0} ## we won't use performance variable for now. 

    #di = DataInclusion(per_ensemble)
    #mat,inds,seeds = di.make_mat()
    #ensembleparams,frames = mat.shape
    #frame_mapping = {}
    #for f in range(frames): 
    #    including = mat[:,f]
    #    excluding = ~mat[:,f]
    #    params_including = np.array(seeds)[including]
    #    params_excluding = np.array(seeds)[excluding]
    #    frame_mapping[f] = {"include":params_including,"exclude":params_excluding}

    ### Step 2: Now calculate mean deviations for including and excluding:
    #splits = {}
    #influences = {}
    #influence_vars = {} ## we can calcualate the delta bias and variance. 
    #influence_ses = {} ## calculate the delta se as well to make things fair. 
    #frameorder = []
    #for frame,fdict in tqdm(frame_mapping.items()):
    #    include = []
    #    exclude = []
    #    for e in ensembles:
    #        idstring = "seed: {}, frames: {}".format(e["seed"],e["frames"])
    #        if idstring in fdict["include"]:    
    #            include.extend(e["modeldiffs"].values()) ## modeldiffs is an array of shape 1000,2,4
    #        elif idstring in fdict["exclude"]: 
    #            exclude.extend(e["modeldiffs"].values())
    #        else:
    #            raise Exception("error in parsing paths! ")
    #    ## We should take the norm, then calculate statistics..     
    #    dist_include = np.linalg.norm(np.array(include),axis = 2)
    #    dist_exclude = np.linalg.norm(np.array(exclude),axis = 2)

    #    mean_include = np.mean(dist_include,axis = 0) ## take the mean along the ensembles dimension   
    #    mean_exclude = np.mean(dist_exclude,axis = 0)    
    #    var_include = np.var(dist_include,axis = 0)    
    #    var_exclude = np.var(dist_exclude,axis = 0)    
    #    se_include = np.std(dist_include,axis = 0)/np.sqrt(len(include))    
    #    se_exclude = np.std(dist_exclude,axis = 0)/np.sqrt(len(exclude))    

    #    influence = mean_include-mean_exclude ## with this measure of influence, higher means the frame is worse: it's the bias
    #    influence_var = var_include-var_exclude ## likewise, higher variance is worse
    #    influence_se = se_include-se_exclude ## same for standard error. 
    #    influences[frame] = influence
    #    influence_ses[frame] = influence_se
    #    influence_vars[frame] = influence_var
    #    ## Finally, save the raw partition: 
    #    splits[frame] = {"include":[gt-i for i in include],"exclude":[gt-e for e in exclude]} ## must be gt - difference bc we calculated as gt - detection before. 
    #arrayrep = np.array([v for v in influences.values()]).squeeze()  ## (of shape nb_frames,1000,2,4)  
    #arrayrep_se = np.array([vse for vse in influence_ses.values()]).squeeze()
    #arrayrep_var = np.array([vse for vse in influence_vars.values()]).squeeze()
    ##normed = np.linalg.norm(arrayrep,axis = 2) ## now as distances in xy

    #fig,ax = plt.subplots(arrayrep.shape[-1],1,figsize = (25,25))
    #for i in range(arrayrep.shape[-1]):
    #    mappable = ax[i].matshow(arrayrep[:,:,i],aspect = "auto")
    #    ax[i].set_yticks(range(len(frame_mapping)))
    #    ax[i].set_yticklabels(inds)
    #    fig.colorbar(mappable,ax=ax[i])
    #plt.title("Marginal change in bias as a function of training frame inclusion")    
    #plt.tight_layout()    
    #plt.savefig(os.path.join(scriptdir,"../","images/","influence_mat"))

    #fig,ax = plt.subplots(arrayrep_var.shape[-1],1,figsize = (25,25))
    #for i in range(arrayrep_var.shape[-1]):
    #    mappable = ax[i].matshow(arrayrep_var[:,:,i],aspect = "auto")
    #    ax[i].set_yticks(range(len(frame_mapping)))
    #    ax[i].set_yticklabels(inds)
    #    fig.colorbar(mappable,ax=ax[i])
    #plt.title("Marginal change in variance as a function of training frame inclusion")    
    #plt.tight_layout()    
    #plt.savefig(os.path.join(scriptdir,"../","images/","influence_mat_var"))

    #fig,ax = plt.subplots(arrayrep_se.shape[-1],1,figsize = (25,25))
    #for i in range(arrayrep_se.shape[-1]):
    #    mappable = ax[i].matshow(arrayrep_se[:,:,i],aspect = "auto")
    #    ax[i].set_yticks(range(len(frame_mapping)))
    #    ax[i].set_yticklabels(inds)
    #    fig.colorbar(mappable,ax=ax[i])
    #plt.title("Marginal change in standard error as a function of training frame inclusion")    
    #plt.tight_layout()    
    #plt.savefig(os.path.join(scriptdir,"../","images/","influence_mat_se"))

    #joblib.dump({"frame_index":inds,"delta_biases":arrayrep,"delta_variances":arrayrep_var,"delta_ses":arrayrep_se,"raw_data":splits},os.path.join(scriptdir,"script_outputs","influence_data"))

if __name__ == "__main__":
    main()
