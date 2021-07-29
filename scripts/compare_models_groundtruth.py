## compare models to the groundtruth detection. 
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
import dgp_ensembletools.models as models
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deepgraphpose.models.eval import load_pose_from_dlc_to_dict
import click
import matplotlib.pyplot as plt

base_dir = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21/"
ensemble_template = "job__ensemble-dgp_dgpreal_{f}_seed{s}"
ensemble_template2 = "job__ensemble-dgp_dgpreal2_{f}_seed{s}" ## for other seed
model_template = "ensemble-model{n}-2030-01-0{n}"
data_directory = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21" ## redundant with ensemblepath below. 
ensemblepath = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21"
task = "ibl1"

colors = {2:"#a50026",27:"#d73027",42:"#f46d43",52:"#fee090",89:"#e0f3f8",102:"#abd9e9",122:"#74add1",142:"#4575b4",162:"#fdae61",7700:"#313695"}
labels = {5:"10%",17:"30%"}
markers = ["o","x"] ## o for no outliers, x for outliers
label = ["no outlier","outlier"]

# determine if contains outliers
def determine_outliers(labellist,seed,nb_frames):
    """Determine if a given seed and frame count contains outliers

    :param labellist: path to ordered list of labels with indication of if they contain outliers or not.  
    :param seed: seed to use to calculate outlier presence. 
    :param nb_frames: number of frames used to calculate outlier presence

    :return: bool
    """
    ## First get the list:
    with open(labellist,"rb") as fp:
        labels = pickle.load(fp)
    nb_datapoints = len(labels)
    ## TODO here get a simple indicator of if the element is an outlier or not.
    is_outlier = lambda x: sum(map(float,x[1:]))>0
    outliers = np.array(list(map(is_outlier,labels))) ## boolean list of if outlier or not.

    seeddict = {}

    si = int(seed)
    np.random.seed(si)
    sequence = np.sort(np.random.choice(nb_datapoints,nb_frames,replace=False))
    np.random.seed()
    ## determine if they contain outliers.
    outliers = np.any(outliers[sequence]) ## True or False
    return outliers

# parse modelnames into dictionary of relevant info
def parse_modelname(string,labellist,ensemblesfolder):
    """Given a string, parse out the seed and number of frames used to train the model if it corresponds to a model. 

    :param string: string to parse to check if it's a model name
    :param labellist: list of labeled data and categorization as outlier or no.  
    :param ensemblesfolder: the base folder where ensembles are stored. 
    :return: a dictionary containing the full name, string, and frame id (if real). None otherwise.
    """
    ## We need to account for two different prefixes now. 
    split_ens_temp = ensemble_template.split("{f}")
    split_ens_temp2 = ensemble_template2.split("{f}")
    template_prefix = split_ens_temp[0]
    template_prefix2 = split_ens_temp2[0]

    template_seedind = split_ens_temp[1].split("{s}")[0]
    if string.startswith(template_prefix): ## TODO or other prefix
        frames,seed = string.split(template_prefix)[-1].split(template_seedind)
        return {"name":string,
                "frames":int(frames),
                "seed":int(seed),
                "template":ensemble_template,
                "outliers":determine_outliers(labellist,int(seed),int(frames)),
                "models":[models.TrainedModel(os.path.join(ensemblesfolder,string,"process_results",model_template.format(n = n+1)),"avi") for n in range(2)]
                }
    elif string.startswith(template_prefix2): 
        frames,seed = string.split(template_prefix2)[-1].split(template_seedind)
        return {"name":string,
                "frames":int(frames),
                "seed":int(seed),
                "template":ensemble_template2,
                "outliers":determine_outliers(labellist,int(seed),int(frames)),
                "models":[models.TrainedModel(os.path.join(ensemblesfolder,string,"process_results",model_template.format(n = n+1)),"avi") for n in range(2)]
                }
    else:    
        return None

# get training frames: 
def get_training_frames(nb_frames,seeds,data_directory,ensemblepath,task,shuffle = 1,trainingsetindex = 0):
    """Get a list of training frames corresponding to a model. 
    :param nb_frames: number of frames to evaluate on. 
    :param seeds: seeds to evaluate 
    :param data_directory: root directory where all ensembles are stored. 
    :param ensemblepath: path to the ensemble. 
    :param task: video name 
    :return: a list of frame indices indicating the training data. 
    """
    traces_folder = "videos_pred/{}_labeled.csv".format(task)

    for seed_idx, seed in enumerate(seeds):
        #date = "2050-{:02d}-{:02d}".format(date_idx + 1, seed)
        cfg_folder = str(Path(data_directory)/ "{}".format(ensemblepath)/"job__ensemble-dgp_dgpreal_{}_seed{}".format(nb_frames,seed)/"process_results/ensemble-model1-2030-01-01")
        cfg_yaml = str(Path(cfg_folder)/ "config.yaml")
        cfg = auxiliaryfunctions.read_config(cfg_yaml)
        cfg["project_path"] = os.path.join(data_directory,ensemblepath,str(seed))
        label_file = str(Path(cfg_folder)/ traces_folder)
        #print(label_file)
        try:
            labels = load_pose_from_dlc_to_dict(label_file)
            xr_dgp = labels['x'] # T x D
            yr_dgp = labels['y']
        except:
            skipped_seed[date_idx].append(seed_idx)
            continue
        # ------
        # get train/test sets (indices for video)
        trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
        datafn, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(
            trainingsetfolder, cfg['TrainingFraction'][trainingsetindex], shuffle, cfg)
        # Load meta data
        data, trainIndices, testIndices, trainFraction = auxiliaryfunctions.LoadMetadata(
            os.path.join(cfg_folder, metadatafn))
        # the train indices are the indices of the labeled frames? or the video itself
        # they are of the training set, so read the indices of the video
        Data = pd.read_hdf(os.path.join(cfg_folder, str(trainingsetfolder),
                                        'CollectedData_' + cfg["scorer"] + '.h5'),
                           'df_with_missing') #* dlc_cfg['global_scale']
        num_labels, _ = Data.values.shape
        labeled_frames = np.empty(num_labels).astype('int')
        for frame_idx in range(num_labels):
            idx_name = int(Path(Data.iloc[frame_idx].name).stem[3:])
            labeled_frames[frame_idx] = idx_name
        # Video frames user for training and testing:
        train_frames = labeled_frames[trainIndices]
        test_frames = labeled_frames[testIndices]
        return train_frames

@click.command(help = "Compare individual model output to groundtruth. Assumes that ensembles are grouped by seed and number of training frames, with two models per ensemble.")
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--groundtruth",help = "path to groundtruth .mat file of hand labeled data.",default = "../data/ibl/ibl1_true_xy_all_918pm.mat")
@click.option("--ensemblesfolder",help = "base folder where all ensemble folders are kept (holds folders titled `job__ensemble-dgp_dgpreal_{frames}_seed{seed}`)",default = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21/")
def main(labellist,groundtruth,ensemblesfolder):
    """Take a collection of ensembes of trained networks, and calculate the losses on a set of test data. 

    """
    ## Get all ensemble specifying info and calculate if each ensemble has seen outliers or not: 
    ensembles = [parse_modelname(m,labellist,ensemblesfolder) for m in os.listdir(ensemblesfolder) if parse_modelname(m,labellist,ensemblesfolder) is not None]

    ## Next, figure out the union of all the training data.  
    all_frames = []
    for e in ensembles:
        nb_frames = e["frames"]
        seeds = e["seed"]
        template = e["template"]## per- run template. because we prefix some runs as dgpreal2
        train_frames = get_training_frames(nb_frames,[seeds],data_directory,ensemblepath,task)
        ## Quick check: assert that these frames are in the appropriate model folder: 
        for t in train_frames:
            datafolder = os.path.join(base_dir,template.format(f =nb_frames, s = seeds),"process_results",model_template.format(n=1),"labeled-data",task)
            contents = os.listdir(datafolder)
            frame_id = "img{0:03d}.png".format(t)
            assert frame_id in contents, "Mismatch between extracted training data and data found in labeled data folder"
        all_frames.extend(train_frames)

    all_training_frames = list(set(all_frames))
    ## Next, evaluate each network's performance on the complement of the union of all the training data.
    all_test_frames = np.setdiff1d(np.arange(1000),train_frames).astype('int')
    for e in ensembles:
        e["modelrmses"] = {"model1":None,"model2":None}
        for mi,m in enumerate(e["models"]):
            rmse = m.compare_groundtruth("{}_labeled.mp4".format(task),
                    groundtruth,
                    partperm = [1,3,0,2],
                    indices = all_test_frames) ## the parts are in the wrong order. fix them when comparing. 
            e["modelrmses"]["model{}".format(mi+1)] = rmse

    ## Plot all performances. 
    fig,ax = plt.subplots()
    for e in ensembles:
        frames = e["frames"]
        outliers = e["outliers"]
        seed = e["seed"]
        for mi,(m,rmse) in enumerate(e["modelrmses"].items()):
            if frames in (5,17) and seed == 2 and mi == 1:
                ax.scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 3,label = label[outliers])
            else:    
                ax.scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 3)
        ax.set_xticks([5,17])    
        plt.xlabel("number of training frames")
        plt.ylabel("rmse")
        plt.legend()    
    plt.savefig("../images/compare_models_groundtruth_output.png")        
    plt.close()
    ## We can also plot them split: 
    fig,ax = plt.subplots(1,2)
    for e in ensembles:
        frames = e["frames"]
        outliers = e["outliers"]
        seed = e["seed"]
        ## did the 5 frame distribution contain outliers? 
        any_outliers = all([determine_outliers(labellist,seed,5),determine_outliers(labellist,seed,17)]) ## if there are no outliers, it's always going to be the 5 percent distribution
        for mi,(m,rmse) in enumerate(e["modelrmses"].items()):
            if frames in (5,17) and seed == 2 and mi == 1:
                ax[int(any_outliers)].scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 2,label = label[outliers])
            else:    
                ax[int(any_outliers)].scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 2)
        [axi.set_xticks([5,17]) for axi in ax]   
    plt.legend()    
    ax[0].set_xlabel("number of training frames")
    ax[1].set_xlabel("number of training frames")
    ax[0].set_ylabel("rmse")
    plt.savefig("../images/compare_models_groundtruth_output_split.png")        
    plt.close()

        


if __name__ == "__main__":
    main()


