## Assume that we have a directory of json files that give the individual model and consensus outputs. 
import click
from compare_models_groundtruth import ensemble_template,determine_outliers
import numpy as np
import json
import matplotlib.pyplot as plt
import os

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
    template_prefix = split_ens_temp[0]

    template_seedind = split_ens_temp[1].split("{s}")[0]
    if string.startswith(template_prefix): ## TODO or other prefix
        frames,seedext = string.split(template_prefix)[-1].split(template_seedind)
        seed=seedext.split("results.json")[0]
        return {"name":string,
                "frames":int(frames),
                "seed":int(seed),
                "template":ensemble_template,
                "outliers":determine_outliers(labellist,int(seed),int(frames)),
                }

@click.command(help = "Compare individual model and consensus output to groundtruth. Assumes that ensembles are grouped by seed and number of training frames, into json output with four models per ensemble.")
@click.option("--labellist",help = "path to pickled list of labels and indicator of if they are outliers or not.",default = "../data/ibl/ordered_classified_list")
@click.option("--ensemblesfolder",help = "base folder where all ensemble folders are kept (holds folders titled `job__ensemble-dgp_dgpreal_{frames}_seed{seed}`)",default = "/Volumes/TOSHIBA EXT STO/pose_results_07_22_21/")
@click.option("--resultsfolder",help="path to folder containing json files of outputs.",default = "../data/ibl/consensus_performance")
def main(labellist,ensemblesfolder,resultsfolder):
    results = os.listdir(resultsfolder)
    ensembles = []
    for f in results:
        ensembledata = parse_modelname(f,labellist,ensemblesfolder)
        with open(os.path.join(resultsfolder,f)) as f:
            performance = json.load(f)
        ensembledata["performance"] = performance    
        ensembles.append(ensembledata)

        
    colors = {2:"#a50026",27:"#d73027",42:"#f46d43",52:"#fee090",89:"#e0f3f8",102:"#abd9e9",122:"#74add1",142:"#4575b4",162:"#fdae61",7700:"#313695"}
    labels = {5:"10%",17:"30%"}
    markers = ["o","x"] ## o for no outliers, x for outliers
    label = ["no outlier","outlier"]
    fig,ax = plt.subplots()
    all_seeds = {c:{5:None,17:None} for c in colors}
    outliersdict = {c:{5:None,17:None} for c in colors}
    for e in ensembles:
        frames = e["frames"]
        outliers = e["outliers"]
        seed = e["seed"]
        #for mi,(m,rmse) in enumerate(e["modelrmses"].items()):
            #if frames in (5,17) and seed == 2 and mi == 1:
            #    ax.scatter(frames+np.random.randn(),rmse,s=50,marker = markers[outliers],color= colors[seed],linewidth = 2,label = label[outliers],alpha = 0.5)
            #else:    
            #    ax.scatter(frames+np.random.randn(),rmse,s=50,marker = markers[outliers],color= colors[seed],linewidth = 2,alpha = 0.5)
        median_performance = e["performance"]["median"]    
        print(median_performance)
        all_seeds[seed][frames] = median_performance
        outliersdict[seed][frames] = outliers

    for s in all_seeds:    
        vals = [all_seeds[s][5],all_seeds[s][17]]
        x = np.array([5,17])# +np.random.randn(2,)
        print(x,vals)
        ax.plot(x,vals,marker = markers[outliersdict[s][5]] ,color= colors[s],linewidth = 2)


        #ax.scatter(frames+np.random.randn(),median_performance,s=50,marker =outliersdict[s][5] ,color= colors[s],linewidth = 2)
        ax.set_xticks([5,17])    
        plt.xlabel("number of training frames")
        plt.ylabel("rmse")
        plt.legend()    
    plt.savefig("../images/compare_models_groundtruth_output_consensus.png")        
    plt.close()

if __name__ == "__main__":    
    main()
