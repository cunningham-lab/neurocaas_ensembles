## Look at individual subsets of training data, and make a matrix representation of what datapoints are included or not included. 
import click
import os 
import json
import numpy as np
import matplotlib.pyplot as plt
from dgp_ensembletools.models import Ensemble
from pose_aum.datavis import DataInclusion
from compare_models_groundtruth import  get_training_frames
from plot_consensus import parse_modelname

scriptdir = os.path.abspath(os.path.dirname(__file__))

modelpaths = "ensemble-model{i}-2030-01-0{i}"

@click.command("make matrix of data describing which models contain which datapoints.") 
@click.option("--video-name",default = "ibl1_labeled.mp4")
@click.option("--groundtruth",default = "../data/ibl/ibl1_true_xy_all_918pm.mat")
@click.option("--partperm",default = "ibl",help="if we need to permute the labels of the groundtruth before comparing.")
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--basefolder",default = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21")
@click.option("--resultsfolder",help="path to folder containing json files of outputs.",default = "../data/ibl/consensus_performance")
def main(video_name,groundtruth,partperm,labellist,basefolder,resultsfolder):
    """Runs after calculate_consensus script. Takes the performance estimates and ensembles from that data, and determines properties of the data from them.  

    """
    if partperm == "ibl":
        partperm = np.array([1,3,0,2]) # permute parts before comparing
    else:
        partperm = None
    print("got here")
    #ensembles = [parse_modelname(m,labellist,basefolder) for m in os.listdir(basefolder) if parse_modelname(m,labellist,basefolder) is not None]

    results = os.listdir(resultsfolder)
    ensembles = []
    for f in results:
        ensembledata = parse_modelname(f,labellist,basefolder)
        with open(os.path.join(resultsfolder,f)) as f:
            performance = json.load(f)
        ensembledata["performance"] = performance
        ensembles.append(ensembledata)


    ## Next, figure out the union of all the training data.  
    all_frames = []
    per_ensemble = {}
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
        per_ensemble["seed: {}, frames: {}".format(seeds,nb_frames)] = {"frame_inds":train_frames,"performance":e["performance"]["median"]}

    all_training_frames = list(set(all_frames))
    di = DataInclusion(per_ensemble)
    ## plot data matrix:
    mat,inds,seeds = di.make_mat()
    plt.matshow(mat)
    plt.xticks(range(len(inds)),inds,rotation = 90)
    plt.yticks(range(len(seeds)),seeds)
    plt.xlabel("frame indices")
    plt.ylabel("ensemble seeds")
    plt.tight_layout()
    plt.title("Datapoints included in different ensembles")
    plt.savefig(os.path.join(scriptdir,"../","images","DataInclusionMatrix{}.png".format(video_name)))
    ## plot sorted data matrix:
    mat,inds,seeds = di.make_sorted_mat()
    plt.matshow(mat)
    plt.xticks(range(len(inds)),inds,rotation = 90)
    plt.yticks(range(len(seeds)),seeds)
    plt.xlabel("frame indices")
    plt.ylabel("ensemble seeds")
    plt.title("Datapoints included in different ensembles (sorted best to worst)")
    plt.tight_layout()
    plt.savefig(os.path.join(scriptdir,"../","images","DataInclusionMatrix_Sorted{}.png".format(video_name)))

if __name__ == "__main__":
    main()
