## Works on REMOTE AWS INSTANCE
# Calculates the consensus detection for ensembles of size four (across two different runs of july data.)
import click
import os 
import json
import numpy as np
from dgp_ensembletools.models import Ensemble
from compare_models_groundtruth import parse_modelname, get_training_frames
ensemble_template_prefix = "job__ensemble-dgp_dgpreal_"
ensemble_template_prefix2 = "job__ensemble-dgp_dgpreal2_"
templates = [ensemble_template_prefix,ensemble_template_prefix2]
modelpaths = "ensemble-model{i}-2030-01-0{i}"

def make_full_ensemble(path,ensemblefolder):
    """Given a prefix path that matches the ensemble template prefix, make an ensemble from it. 
    """
    suffix = path.split(ensemble_template_prefix)[-1]
    try:
        paths = [os.path.join(ensemblefolder,ep+suffix,"process_results",modelpaths.format(i=i)) for ep in templates for i in [1,2]] 
        ensemble = Ensemble(os.path.join(ensemblefolder,path,"process_results"),paths,ext = "mp4")
    except AssertionError: ## for the 70 and 90% ensembles, we only have one round:     
        paths = [os.path.join(ensemblefolder,ep+suffix,"process_results",modelpaths.format(i=i)) for ep in [ensemble_template_prefix] for i in [1,2]] 
        ensemble = Ensemble(os.path.join(ensemblefolder,path,"process_results"),paths,ext = "mp4")

    return ensemble


def get_rmse(ensemble,videoname, groundtruth,partperm,test_frames):    
    """Given an ensemble, calculates the rmse for it. 

    :return: dictionary of individual and median performance. 
    """
    return ensemble.compare_groundtruth(videoname,groundtruth,partperm =partperm,indices = test_frames)


@click.command("get the consensus output")
@click.option("--video-name",default = "ibl1_labeled.mp4")
@click.option("--groundtruth",default = "/home/ubuntu/pose_aum/data/ibl/ibl1_true_xy_all_918pm.mat")
@click.option("--partperm",default = "ibl",help="if we need to permute the labels of the groundtruth before comparing.")
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--basefolder",default = "/home/ubuntu/july_data/",type = click.Path(resolve_path = True))
@click.option("--resultsfolder",help="path where we will write result.json files",default = "../data/ibl/consensus_performance",type = click.Path(resolve_path= True))
def main(video_name,groundtruth,partperm,labellist,basefolder,resultsfolder):
    """Runs creation and rmse retrieval of ensembles.  

    """
    if partperm == "ibl":
        partperm = np.array([1,3,0,2]) # permute parts before comparing
    else:
        partperm = None
    print("got here")
    ensembles = [parse_modelname(m,labellist,basefolder) for m in os.listdir(basefolder) if parse_modelname(m,labellist,basefolder) is not None]

    ## Next, figure out the union of all the training data.  
    all_frames = []
    for e in ensembles:
        nb_frames = e["frames"]
        seeds = e["seed"]
        template = e["template"]## per- run template. because we prefix some runs as dgpreal2
        train_frames = get_training_frames(nb_frames,[seeds],basefolder,basefolder,video_name.split("_labeled.mp4")[0])
        ## Quick check: assert that these frames are in the appropriate model folder: 
        for t in train_frames:
            datafolder = os.path.join(basefolder,template.format(f =nb_frames, s = seeds),"process_results",modelpaths.format(i=1),"labeled-data",video_name.split("_labeled.mp4")[0])
            contents = os.listdir(datafolder)
            frame_id = "img{0:03d}.png".format(t)
            assert frame_id in contents, "Mismatch between extracted training data and data found in labeled data folder"
        all_frames.extend(train_frames)

    all_training_frames = list(set(all_frames))

    ## Next, evaluate each network's performance on the complement of the union of all the training data.
    all_test_frames = np.setdiff1d(np.arange(1000),train_frames).astype('int')

    ## Now create full ensembles
    all_ensembles = {} 
    all_folders = os.listdir(basefolder)
    for f in all_folders:
        if f.startswith(ensemble_template_prefix):
            e = make_full_ensemble(f,basefolder)
            rmse_dict = get_rmse(e,video_name,groundtruth,partperm,all_test_frames)
            rmse_dict["name"] = f
            with open(os.path.join(resultsfolder,f+"results.json"),"w") as f:
                json.dump(rmse_dict,f)

if __name__ == "__main__":
    main()


