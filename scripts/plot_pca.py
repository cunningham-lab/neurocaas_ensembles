import os
import joblib 
import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from compare_models_groundtruth import colors,markers

here = os.path.abspath(os.path.dirname(__file__))

def get_plottinginds(labels):
    """Given a list of dictionaries containing data about the plotting labels, returns three separate lists along which they can be classified: seed and number of frames. 

    """
    seedfunc = lambda x: x["seed"]
    framefunc = lambda x: x["frames"]
    outlierfunc = lambda x: x["outliers"]
    seeds = map(seedfunc,labels)
    frames = map(framefunc,labels)
    outliers = map(outlierfunc,labels)
    return seeds,frames,outliers

@click.command(help = "plot the results of a created pca model")
@click.option("--modelpath",type = click.Path(file_okay = True, dir_okay = False, exists = True),help = "path to a dictionary pickled via joblib with two fields: `labels`, containing a flat index of each sample in the pca model's metadata, `transformed`, containing the transformed pca loading weights, and `model`, containing the sklearn model itself.")
@click.option("--nb-parts",type = click.INT,help = "number of animal parts to track",default = 4)
def main(modelpath,nb_parts):
    """The image we will plot here will be like the PCA plot in the "Motivation" section of the document. 

    """
    pca = joblib.load(modelpath)
    labels = pca["labels"]
    model = pca["model"]
    transformed = pca["transformed"]
    evr = model.explained_variance_ratio_
    components = model.components_
    pc0 = components[0,:].reshape((-1,2,nb_parts))
    pc1 = components[1,:].reshape((-1,2,nb_parts))
    mean = model.transform(np.zeros((1,np.prod(pc0.shape))))
    fig,ax = plt.subplots(2,2,figsize = (10,10))
    ax[0,0].bar(range(len(evr[:5])),evr[:5])
    ax[0,1].plot(pc0[:,0,0],label = "PC 0 X coord")
    ax[0,1].plot(pc0[:,1,0],label = "PC 0 Y coord")
    ax[1,1].plot(pc1[:,0,0],label = "PC 1 X coord")
    ax[1,1].plot(pc1[:,1,0],label = "PC 1 Y coord")
    ax[0,1].legend()
    ax[1,1].legend()
    seeds, frames, outliers = get_plottinginds(labels)
    params = zip(seeds,frames,outliers)
    for pi,(seed,frame,outliers) in enumerate(params):
        ax[1,0].scatter(transformed[pi,0],transformed[pi,1],color = colors[seed],s = 2*frame,marker = markers[outliers])
        #ax[1,1].scatter(transformed[pi,2],transformed[pi,3],color = colors[seed],s = 2*frame,marker = markers[outliers])
    ax[1,0].scatter(mean[0,0],mean[0,1],marker = "X",s = 50,color = "black")
    #ax[1,1].scatter(mean[0,2],mean[0,3],marker = "X",s = 50,color = "black")

    ax[0,0].set_title("Explained Variance Ratio: \n Top 5 PCs")
    ax[0,0].set_ylabel("Variance Ratio")
    ax[0,0].set_xlabel("PC")
    
    ax[0,1].set_title("Error Reconstruction: PC 0 (paw)")
    ax[0,1].set_ylabel("PC Component Value")
    ax[0,1].set_xlabel("Frame")

    ax[1,1].set_title("Error Reconstruction: PC 1 (paw)")
    ax[1,1].set_ylabel("PC Component Value")
    ax[1,1].set_xlabel("Frame")

    ax[1,0].set_title("Error Space: PC 0 vs. PC 1")
    ax[1,0].set_xlabel("PC 0 loading weight")
    ax[1,0].set_ylabel("PC 1 loading weight")

    legend_elements = [Line2D([0],[0],marker = "X",markersize = 10,color = "w",markerfacecolor = "black",label = "Groundtruth"),
            Line2D([0],[0],marker = markers[0],color = "w",markerfacecolor = colors[2],markersize = 10,label = "No Outlier"),
            Line2D([0],[0],marker = "X",color = "w",markerfacecolor = colors[2],markersize = 10,label = "Outlier")]
    ax[1,0].legend(handles = legend_elements)
    #ax[1,1].set_xlabel("PC 2 loading weight")
    #ax[1,1].set_ylabel("PC 3 loading weight")
    plt.tight_layout()


    plt.savefig(os.path.join(here,"../","images","pcafig"))

if __name__ == "__main__":
    main()

