## Analyze the influence estimates you obtained previously. 
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import numpy as np
import click
import os 
import joblib

here = os.path.abspath(os.path.dirname(__file__))
legend_elements = [Line2D([0],[0],marker = "X",markersize = 10,color = "w",markerfacecolor = "red",label = "Not included"),
        Line2D([0],[0],marker = "o",color = "w",markerfacecolor = "blue",markersize = 10,label = "Included")]

def plot_memorization(frameind,framedir,biases,std_errors,raw_data,data_id):
    """Plots the memorization statistics for a particular training frame.

    :param frameind: integer index of the frame- should be the index in the original training set so we can retrieve the corresponding frame. 
    :param framedir: directory where we expect to get frames. 
    :param biases: array of shape (parts,), giving the change in biases for each part. 
    :param std_errors: array of shape (std_errors,) giving the change in standard error for each part. 
    :param raw_data: dictionary with two fields: include and exclude, with each containing lists of full time series of raw data giving the xy positions of different body parts.  
    :param data_id: id string to attach to data. 
    """
    framepath = os.path.join(framedir,"img{0:03d}.png".format(frameind))
    #frame = mpimg(framepath)
    img = plt.imread(framepath)
    fig,ax = plt.subplots(1,4,figsize = (15,5))
    for i in range(4):
        ax[i].imshow(img)
        for exc in raw_data["exclude"]:
            ax[i].plot(*exc[frameind,:,i],"x",color = "red",alpha = 0.5)
        for inc in raw_data["include"]:
            ax[i].plot(*inc[frameind,:,i],"o",color = "blue",alpha = 0.5)
        ax[i].set_title("$\Delta$ Bias: {:.2f},\n $\Delta$ SE: {:.2f}".format(biases[i],std_errors[i]))
    ax[0].legend(handles = legend_elements)    
    plt.suptitle("Memorization for Training Frame {}".format(frameind))
    plt.savefig(os.path.join(here,"./script_outputs/memorization{}_{}.png".format(frameind,data_id)))
    plt.close()

def plot_influence(trainind,infind,framedir,video,biases,std_errors,raw_data,data_id):
    """Plots influence of one training frame on another arbitrary frame. 
    :param trainind: integer index of the training- should be the index in the original training set so we can retrieve the corresponding frame. 
    :param infind: integer index influenced frame. 
    :param framedir: directory where we expect to get frames. 
    :param video: VideoFileClip object. 
    :param biases: array of shape (parts,), giving the change in biases for each part. 
    :param std_errors: array of shape (std_errors,) giving the change in standard error for each part. 
    :param raw_data: dictionary with two fields: include and exclude, with each containing lists of full time series of raw data giving the xy positions of different body parts.  
    :param data_id: id string to attach to data. 
    """
    frametemp = "img{0:03d}.png"
    trainframepath = os.path.join(framedir,frametemp.format(trainind))
    #infframepath = os.path.join(framedir,frametemp.format(infind))
    #frame = mpimg(framepath)
    trainimg = plt.imread(trainframepath)
    #infimg = plt.imread(infframepath)
    frameindex = infind/video.fps
    print(frameindex,infind,video.fps,video.duration)
    infimg = video.get_frame(frameindex)
    fig,ax = plt.subplots(1,5,figsize = (20,5))
    ax[0].imshow(trainimg)
    ax[0].set_title("Training Frame {}".format(trainind))
    for i in range(4):
        ax[i+1].imshow(infimg)
        for exc in raw_data["exclude"]:
            ax[i+1].plot(*exc[infind,:,i],"x",color = "red",alpha = 0.5)
        for inc in raw_data["include"]:
            ax[i+1].plot(*inc[infind,:,i],"o",color = "blue",alpha = 0.5)
        ax[i+1].set_title("Frame {}, LH Finger {} \n $\Delta$ Bias: {:.2f},\n $\Delta se$: {:.2f}".format(infind,i,biases[i],std_errors[i]))
    ax[1].legend(handles = legend_elements)    
    plt.suptitle("Influence of Training Frame {} on Frame {} Detections".format(trainind,infind))
    line = Line2D([0.263,0.263],[0,1],transform = fig.transFigure,color = "black")
    fig.add_artist(line)
    plt.savefig(os.path.join(here,"./script_outputs/influence{}_{}_{}.png".format(trainind,infind,data_id)))
    plt.close()


@click.command(help = "calculate memorization values and high influence pairs in terms of bias and variance change")
@click.option("--ensembledict",help = "Joblib pickled dictionary that is the output of estimate_influence containing the raw data, frame indices, and (bias, variance, and standard error changes) in array representation.",default = os.path.join(here,"script_outputs","influence_data"))
@click.option("--framedir",help="Directory where all training frames are stored.",default = os.path.join(here,"../","data","ibl","all_frames"))
@click.option("--videopath",help="Path to video clip",default = os.path.join(here,"../","data","ibl_data","1","videos","ibl1.mp4"))
def main(ensembledict,framedir,videopath):

    data_id = ensembledict.split("influence_data_")[-1]
    # Load the data: 
    datadict= joblib.load(ensembledict)
    frames = datadict["frame_index"]
    raw_data = datadict["raw_data"] ## an indexed collection of dictionaries, each of which has the deviation off of the groundtruth for ensembles where the frame in question is included or excluded. 
    biases = datadict["delta_biases"]
    variances = datadict["delta_variances"]
    std_errors = datadict["delta_ses"]
    vfc = VideoFileClip(videopath)

    # First, memorization. For each training frame, calculate how much inclusion of the frame itself influences the bias and standard error of its estimate. Show the actual estimates.  
    for fi,orig_index in enumerate(frames):

        plot_memorization(orig_index,framedir,biases[fi,orig_index,:],std_errors[fi,orig_index,:],raw_data[fi],data_id)

    # Next, influence. For each training frame, calculate its average effect on bias across all frames. Take the highest and lowest effect frames across body parts, and show the frames for which they affect classification. You should show the new predictions as well.  
    
    for fi,orig_index in enumerate(frames):
        avg_effect = np.mean(biases[fi,:,:]) ## 1000,4
        
        if avg_effect > 0:
            intloc = np.unravel_index(np.argmax(biases[fi,:,:],axis = None),biases[fi,:,:].shape)
            print(intloc,"intloc")
            intframe = intloc[0]
        else:    
            intloc = np.unravel_index(np.argmin(biases[fi,:,:],axis = None),biases[fi,:,:].shape)
            print(intloc,"intloc")
            intframe = intloc[0]
        plot_influence(orig_index,intframe,framedir,vfc,biases[fi,intframe,:],std_errors[fi,intframe,:],raw_data[fi],data_id)




if __name__ == "__main__":    
    main()
