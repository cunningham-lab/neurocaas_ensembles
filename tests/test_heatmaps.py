## Specific module to test heatmap outputs. 
import joblib
import os 
import pickle
import numpy as np
import pytest
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))

@pytest.fixture
def get_data():
    """Influence data is organized as a list of length 40 (one for each ensemble), where each element of the list is a dict with keys:  
        - name: name of the ensemble. 
        - frames: number of frames in train set.  
        - seed: seed used to select frames
        - template: template used to format model name. 
        - outliers: whether or not this ensemble contains outliers. 
        - models: the full trained models 
        - modeldiffs: the actual confidences recorded on the full video. 

    """
    inf = joblib.load(os.path.join(here,"../","scripts","script_outputs","confidence_data"))
    return inf

def get_frames(seed, frames, labellist = os.path.join(here,"../","data","ibl","classified_list"), frames_total = 58):
    """Given a seed, number of frames, frame ordering, and total number of frames, returns the indices of frames in the test set.  

    :param seed: random seed for np.random.
    :param frames: number of frames to select
    :param labellist: path to label ordering list. 
    :param frames_total: the total number of available frames. 
    """
    with open(labellist,"rb") as fp:
        labels = pickle.load(fp)
    label_index = [int(l[0].split("img")[-1].split(".png")[0]) for l in labels]    
    si = int(seed)
    np.random.seed(si)
    sequence = np.sort(np.random.choice(frames_total,frames,replace=False))
    output = np.array(label_index)[sequence]
    return output

def test_memorized(get_data):
    """each ensemble should perform with very high confidence on its own training set. 

    """
    total_frames = 58
    all_gt_confs = []
    for m in get_data:
        frameindices = get_frames(m["seed"],m["frames"],frames_total = total_frames)
        for model,mconf in m["modeldiffs"].items():
            plt.plot(mconf[:,0])
            [plt.axvline(x = fi,linestyle = "--",color = "black") for fi in frameindices]
            plt.savefig(os.path.join(here,"test_outputs","{}_{}trainindices.png".format(m["name"],model)))
            plt.close()
            train_frame_performance = mconf[np.array(frameindices)]
            all_gt_confs.extend(train_frame_performance.flatten())

    plt.hist(all_gt_confs,bins = 20)    
    plt.savefig(os.path.join(here,"test_outputs","trainset_confidences.png"))

