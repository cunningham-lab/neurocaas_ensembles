## create pca plot  
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
import dgp_ensembletools.models as models
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deepgraphpose.models.eval import load_pose_from_dlc_to_dict
from compare_models_groundtruth import ensemble_template,ensemble_template2,model_template,data_directory,ensemblepath,task,determine_outliers,parse_modelname,get_training_frames 
import click
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
            datafolder = os.path.join(data_directory,template.format(f =nb_frames, s = seeds),"process_results",model_template.format(n=1),"labeled-data",task)
            contents = os.listdir(datafolder)
            frame_id = "img{0:03d}.png".format(t)
            assert frame_id in contents, "Mismatch between extracted training data and data found in labeled data folder"
        all_frames.extend(train_frames)

    all_training_frames = list(set(all_frames))
    ## Next, evaluate each network's performance on the complement of the union of all the training data per frame.
    all_test_frames = np.setdiff1d(np.arange(1000),train_frames).astype('int')
    for e in ensembles:
        e["modeldiffs"] = {"model1":None,"model2":None}
        for mi,m in enumerate(e["models"]):
            diff = m.compare_groundtruth_pointwise("{}_labeled.mp4".format(task),
                    groundtruth,
                    partperm = [1,3,0,2],
                    indices = all_test_frames) ## the parts are in the wrong order. fix them when comparing. 
            e["modeldiffs"]["model{}".format(mi+1)] = diff

    ## Each diff is an array of shape (~950,2,4) (I forget how many training frames we included)        
    ## We should flatten these, and place all in a single array of shape (80,(950X2X4))
    flat_index = []
    array_prep = []
    for e in ensembles:
        for m,diff in e["modeldiffs"].items():
            flat_index.append({"frames":e["frames"],"seed":e["seed"],"model":m,"outliers":e["outliers"]})
            array_prep.append(diff.flatten())
    full_array = np.array(array_prep)
    print(flat_index)
    pcamodel = PCA()
    fit_vals = pcamodel.fit_transform(full_array)
    print(pcamodel.explained_variance_ratio_)
    model_metadata = {"labels":flat_index,"model":pcamodel,"transformed":fit_vals}
    joblib.dump(model_metadata,"pca_with_labels")

    ### Plot all performances. 
    #colors = {2:"#a50026",27:"#d73027",42:"#f46d43",52:"#fee090",89:"#e0f3f8",102:"#abd9e9",122:"#74add1",142:"#4575b4",162:"#fdae61",7700:"#313695"}
    #labels = {5:"10%",17:"30%"}
    #markers = ["o","x"] ## o for no outliers, x for outliers
    #label = ["no outlier","outlier"]
    #fig,ax = plt.subplots()
    #for e in ensembles:
    #    frames = e["frames"]
    #    outliers = e["outliers"]
    #    seed = e["seed"]
    #    for mi,(m,rmse) in enumerate(e["modelrmses"].items()):
    #        if frames in (5,17) and seed == 2 and mi == 1:
    #            ax.scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 3,label = label[outliers])
    #        else:    
    #            ax.scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 3)
    #    ax.set_xticks([5,17])    
    #    plt.xlabel("number of training frames")
    #    plt.ylabel("rmse")
    #    plt.legend()    
    #plt.savefig("../images/compare_models_groundtruth_output.png")        
    #plt.close()
    ### We can also plot them split: 
    #fig,ax = plt.subplots(1,2)
    #for e in ensembles:
    #    frames = e["frames"]
    #    outliers = e["outliers"]
    #    seed = e["seed"]
    #    ## did the 5 frame distribution contain outliers? 
    #    any_outliers = all([determine_outliers(labellist,seed,5),determine_outliers(labellist,seed,17)]) ## if there are no outliers, it's always going to be the 5 percent distribution
    #    for mi,(m,rmse) in enumerate(e["modelrmses"].items()):
    #        if frames in (5,17) and seed == 2 and mi == 1:
    #            ax[int(any_outliers)].scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 2,label = label[outliers])
    #        else:    
    #            ax[int(any_outliers)].scatter(frames+np.random.randn(),rmse,s=100,marker = markers[outliers],color= colors[seed],linewidth = 2)
    #    [axi.set_xticks([5,17]) for axi in ax]   
    #plt.legend()    
    #ax[0].set_xlabel("number of training frames")
    #ax[1].set_xlabel("number of training frames")
    #ax[0].set_ylabel("rmse")
    #plt.savefig("../images/compare_models_groundtruth_output_split.png")        
    #plt.close()

if __name__ == "__main__":
    main()


