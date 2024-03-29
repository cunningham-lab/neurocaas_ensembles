# Code to handle trained dgp models.  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
os.environ['DLClight'] = 'True'
import sys
#sys.path.insert(0, "/home/ekb2154/data/libraries/dgp_paninski/etc/dgp_tools/")
from deepgraphpose.models.fitdgp_util import get_snapshot_path
import yaml
from deepgraphpose.models.eval import plot_dgp,load_pose_from_dlc_to_dict,setup_dgp_eval_graph
from deepgraphpose.utils_model import get_train_config 
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
from scipy.io import loadmat
from scipy.special import softmax
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import shutil
#matplotlib.use('Agg',warn=False, force=True)
from matplotlib import pyplot as plt
from matplotlib import animation
from moviepy.editor import VideoFileClip,VideoClip
from joblib import Memory
if os.getenv("HOME") == "/Users/taigaabe": 
    location = '/Volumes/TOSHIBA EXT STO/cache'
else:    
    try:
        location = os.path.join(os.getenv("HOME"),"cache")
    except:    
        ## probably running as weird user, no home
        location = "./cache"
try:    
    memory = Memory(location, verbose=0)
except PermissionError:    
    location = "./cache"
    memory = Memory(location, verbose=0)


## Functions to smooth the heatmap detections 
def get_mu_fix(ref_, smooth=True):
    """
    Given an unnoramlized scoremap, returns the 2d softmax 
    :param ref_: should be unnormalized scoremap for outputs of shape (batch size,x,y,part)

    """
    #original thing
    n_frames, d1,d2, nj = ref_.shape
    alpha = make2dgrid_np(ref_[0,:,:,0])
    if smooth:
        softmax_ref = softmax(ref_,(1,2)) ## use the softmax here. 
        ref_y2, ref_x2 = (np.sum(alpha[:,:,:,None,None]*np.transpose(softmax_ref,(1,2,0,3)),(1,2))/np.asarray([d2,d1])[:,None,None]) # 2 , y, x
    else:
        ref_x2 = np.zeros((n_frames, nj))*np.nan#np.empty_like(xr)
        ref_y2 =  np.zeros((n_frames, nj))*np.nan#np.empty_like(yr)
        for nt0 in range(n_frames):
            for njj0 in range(nj):
                ref_y2[nt0,njj0], ref_x2[nt0,njj0] = np.unravel_index(np.argmax(ref_[nt0,:,:,njj0]), ref_.shape[1:3])
    return ref_x2, ref_y2

def make2dgrid_np(ref_c):
    d1, d2 = ref_c.shape
    x_i = np.arange(d1)*d2
    y_i = np.arange(d2)*d1
    ## These are multiplied by the other coordinate in order to ease calculation of the expectation later. 
    xg, yg = np.meshgrid(x_i, y_i)
    alpha = np.array([xg, yg]).swapaxes(1, 2)  # 2 x nx_out x ny_out
    return alpha

#vars: video_name,frame_range,proj_config,shuffle,dgp_model_file,  
def _get_poses_and_heatmap(video_path,frame_range,cfg,proj_config,shuffle,dgp_model_file):
    """Internal function to get poses and heatmap to access cache.

    """
    full_video_clip = get_video_clip(video_path,None)
    n_frames2 = min(frame_range[-1] + 1, np.ceil(full_video_clip.fps * full_video_clip.duration).astype('int'))
    offset=frame_range[0] 
    video_clip = get_video_clip(video_path,range(offset,n_frames2))
    fps = video_clip.fps

    dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
    print('done')
    # %%
    try:
        dlc_cfg.net_type = 'resnet_50'
        sess, mu_n, softmax_tensor, scmap, locref, inputs = setup_dgp_eval_graph(dlc_cfg,
                                                                                 dgp_model_file)
    except:
        dlc_cfg.net_type = 'resnet_101'
        sess, mu_n, softmax_tensor, scmap, locref, inputs = setup_dgp_eval_graph(dlc_cfg,
                                                                                 dgp_model_file)
    nj = dlc_cfg.num_joints
    bodyparts = cfg['bodyparts']
    # %%
    nx, ny = video_clip.size
    nx_out, ny_out = int((nx - dlc_cfg.stride/2)/dlc_cfg.stride  + 5), int((ny - dlc_cfg.stride/2)/dlc_cfg.stride  + 5)
    #%%
    markers = np.zeros((n_frames2 + 1, dlc_cfg.num_joints, 2))
    likes = np.zeros((n_frames2 + 1, dlc_cfg.num_joints))
    softmaxtensor = np.zeros((n_frames2 + 1, ny_out, nx_out, dlc_cfg.num_joints))
    pbar = tqdm(total=n_frames2, desc='processing video frames')
    for ii, frame in enumerate(video_clip.iter_frames()):
        ff = img_as_ubyte(frame)
        mu_n_batch, pred_np1 = sess.run( [mu_n,  scmap], feed_dict={inputs: ff[None, :, :, :]})
        nx_out_true, ny_out_true, _ = pred_np1[0].shape
        markers[ii] = mu_n_batch[0]
        num_frames = mu_n_batch.shape[0]  # nb #+ 1
        sigmoid_pred_np = np.exp(pred_np1) / (np.exp(pred_np1) + 1)        
        #softmaxtensor[ii][:nx_out_true,:ny_out_true] = sigmoid_pred_np[0]
        softmaxtensor[ii][:nx_out_true,:ny_out_true] = pred_np1[0]
        mu_likelihoods = np.zeros((num_frames, nj, 2)).astype('int')
        likelihoods = np.zeros((num_frames, nj))
        offset_mu_jj = 0
        for ff_idx in range(num_frames):
            for jj_idx in range(nj):
                # continuous so pick max in
                mu_jj = mu_n_batch[ff_idx, jj_idx]
                ends_floor = np.floor(mu_jj).astype('int') - offset_mu_jj
                ends_ceil = np.ceil(mu_jj).astype('int') + 1 + offset_mu_jj
                sigmoid_pred_np_jj = sigmoid_pred_np[ff_idx, :, :, jj_idx]
                spred_centered = sigmoid_pred_np_jj[ ends_floor[0]: ends_ceil[0],
                                 ends_floor[1]: ends_ceil[1]]
                mu_likelihoods[ff_idx, jj_idx] = np.unravel_index(
                    np.argmax(spred_centered), spred_centered.shape)
                mu_likelihoods[ff_idx, jj_idx] += [ends_floor[0], ends_floor[1]]
                likelihoods[ff_idx, jj_idx] = sigmoid_pred_np_jj[
                    int(mu_likelihoods[ff_idx, jj_idx][0]), int(mu_likelihoods[ff_idx, jj_idx][1])]
        likes[ii] = likelihoods[0]
        pbar.update(1)

    likes = likes[:ii+1]
    markers = markers[:ii+1]
    softmaxtensor = softmaxtensor[:ii+1,:nx_out_true,:ny_out_true,:]
    pbar.close()
    sess.close()
    video_clip.close()
    print('Finished collecting markers')
    print('\n')
    xx = markers[:, :, 1] * dlc_cfg.stride + 0.5 * dlc_cfg.stride
    yy = markers[:, :, 0] * dlc_cfg.stride + 0.5 * dlc_cfg.stride

    return xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg

def get_video_clip(video_path,frame_range = None):
    full_clip = VideoFileClip(str(video_path))
    if frame_range is None:
        return full_clip
    else:
        assert type(frame_range) == type(range(0,1))
        subclip = full_clip.subclip(frame_range[0]/full_clip.fps,frame_range[-1]/full_clip.fps)
        return subclip

colors = ["red","blue","green","purple","magenta","yellow"]
markers = ['d','s','o','p','*','^','<','>','h']

class Ensemble():
    """Ensemble of trained models. Runs analyses that span a set of models. 
    Initialized with a top level directory, and a list of model directories within that top level directory where individual trained models are assumed to reside. 
    :param topdir: top level directory where we will work with this ensemble.
    :param modeldirs: list of model directory names within topdir (will be parsed as os.path.join(topdir,modeldir[i]))
    :param ext: video file extension we care about.
    :param memory: location of joblib cache. By default, is the location declared at the top of this module. If set to None, will not save to the cache.  

    """
    def __init__(self,topdir,modeldirs,ext,memory = memory):
        self.topdir = topdir
        self.modeldirs = modeldirs
        self.modelpaths = [os.path.abspath(os.path.join(self.topdir,self.modeldirs[i])) for i in range(len(self.modeldirs))]
        for mp in self.modelpaths:
            assert os.path.exists(mp) , f"dir {mp} does not exist."
        self.models = {mi:TrainedModel(mp,ext,memory = memory) for mi,mp in enumerate(self.modelpaths)}
        self.ensembledict = {}

    def get_video_clip(self,video_name,frame_range,modelindex = 0):    
        """Pulls out a subclip of a predicted video. Looks inside a arbitrary model in the ensemble to find the video, assuming they are all the same. 

        :param video_name: name of the video. Assume basename. 
        :param frame_range: the range of frames over which we will predict. a range() object. 
        :param modelindex: an integer indexing into the list of models we have. 
        """
        selected_model = self.models[modelindex]
        assert video_name in selected_model.pred_video_files, "video must have been predicted on."
        video_path = os.path.join(selected_model.project_dir,"videos_pred",video_name)
        full_clip = VideoFileClip(str(video_path))
        subclip = full_clip.subclip(frame_range[0]/full_clip.fps,frame_range[-1]/full_clip.fps)
        return subclip

    def get_poses(self,video_name):
        """Gets the poses for a certain video name. 

        """
        if self.ensembledict.get(video_name,None) is None:
            self.ensembledict[video_name] = {"run{}".format(ind):self.models[ind].get_poses_array(video_name) for ind,model in self.models.items()}
        return self.ensembledict[video_name]

#    def get_scoremaps(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
#        for model in self.models:
#            xx,yy,likes,nj,bodyparts,softmaxtensor,dlc_cfg = model.get_poses_and_heatmap_info(video_name,frame_range,snapshot,shuffle)
#
## Apply the same changes as you did with the median below. 
    #def get_mean_pose(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1,smooth = True):
    #    """Gets the scoremaps across the ensemble for this frame range of this video at this snapshot, and calculates the mean pose from it.  
    #    NOTE: passing frame_range(0,2) will give 1 frame, not 2 as you would expect.  
    #    TODO: Write test for this.  
    #    :param video_name:
    #    :param frame_range:
    #    :param snapshot:
    #    :param shuffle:

    #    """
    #    softmaxtensors = []
    #    for i in range(len(self.models)):
    #        model = self.models[i]
    #        xr,yr,likes,nj,bodyparts,softmaxtensor,dlc_cfg = model.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)
    #        softmaxtensors.append(softmaxtensor)    
    #    ref_ = np.mean(softmaxtensors,0)    
    #    ref_x,ref_y = get_mu_fix(ref_,smooth = True)
    #    #ref_x = np.empty_like(xr)
    #    #ref_y = np.empty_like(yr)
    #    #len_range = len(frame_range)-1 ## just index relative to the subclip. Strangely moviepy returns len(framerange) -1 frames...
    #    #for nt0 in range(len_range):
    #    #    for njj0 in range(nj):
    #    #        ref_y[nt0,njj0], ref_x[nt0,njj0] = np.unravel_index(np.argmax(ref_[nt0,:,:,njj0]), ref_.shape[1:3])
    #    ref_x = ref_x* dlc_cfg.stride + 0.5 * dlc_cfg.stride
    #    ref_y = ref_y* dlc_cfg.stride + 0.5 * dlc_cfg.stride
    #    print("scaling refs.")
    #            
    #    return ref_x,ref_y

    def get_scoremaps(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Get scoremaps (unnormalized likelihoods) from the convnet output. Return these and additionally normalized softmax tensors.

        """
        scmaps = []
        for i in range(len(self.models)):
            model = self.models[i]
            xr,yr,likes,nj,bodyparts,scmap,dlc_cfg = model.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)
            scmaps.append(scmap)
        return scmaps    

    def get_logistic(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """ Get standard logistic transformation of scoremaps. 

        """
        softmax_tensors = []
        scmaps = self.get_scoremaps(video_name,frame_range,snapshot,shuffle)
        for scmap_ in scmaps:
            softmax_tensors.append(np.exp(scmap_)/(np.exp(scmap_)+1))
        return softmax_tensors    

    def get_median_scoremap(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the median scoremap. First collects a group of scoremaps, then takes the softmax of each, applies the median, and then transforms back into the scoremap. This is necessary to then pass this scoremap into the smoothing function.   

        """
        logistic_tensors = self.get_logistic(video_name,frame_range,snapshot,shuffle)
        median_logistic = np.median(logistic_tensors,0)
        median_scoremap = np.log(median_logistic) - np.log(1-median_logistic)
        return median_scoremap

    def get_mean_scoremap(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the median scoremap. First collects a group of scoremaps, then takes the softmax of each, applies the mean, and then transforms back into the scoremap. This is necessary to then pass this scoremap into the smoothing function.   

        """
        logistic_tensors = self.get_logistic(video_name,frame_range,snapshot,shuffle)
        mean_logistic = np.mean(logistic_tensors,0)
        mean_scoremap = np.log(mean_logistic) - np.log(1-mean_logistic)
        return mean_scoremap

    def get_median_pose(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):

        """Gets the scoremaps across the ensemble for this frame range of this video at this snapshot, and calculates the median pose from it.  
        NOTE: passing frame_range(0,2) will give 1 frame, not 2 as you would expect. 
        :param video_name:
        :param frame_range:
        :param snapshot:
        :param shuffle:

        """
        for i in range(len(self.models)):
            model = self.models[i]
            xr,yr,likes,nj,bodyparts,scmap,dlc_cfg = model.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)

        ref_ = self.get_median_scoremap(video_name,frame_range,snapshot,shuffle)

        ref_x,ref_y = get_mu_fix(ref_,smooth = True)

        ref_x = ref_x* dlc_cfg.stride + 0.5 * dlc_cfg.stride
        ref_y = ref_y* dlc_cfg.stride + 0.5 * dlc_cfg.stride
                
        return ref_x,ref_y

    def get_mean_pose(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):

        """Gets the scoremaps across the ensemble for this frame range of this video at this snapshot, and calculates the median pose from it.  
        NOTE: passing frame_range(0,2) will give 1 frame, not 2 as you would expect. 
        :param video_name:
        :param frame_range:
        :param snapshot:
        :param shuffle:
        :return: two arrays, ref_x, ref_y, each of shape (time,parts)

        """
        for i in range(len(self.models)):
            model = self.models[i]
            xr,yr,likes,nj,bodyparts,scmap,dlc_cfg = model.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)

        ref_ = self.get_mean_scoremap(video_name,frame_range,snapshot,shuffle)

        ref_x,ref_y = get_mu_fix(ref_,smooth = True)

        ref_x = ref_x* dlc_cfg.stride + 0.5 * dlc_cfg.stride
        ref_y = ref_y* dlc_cfg.stride + 0.5 * dlc_cfg.stride
        print("scaling refs.")
                
        return ref_x,ref_y

    def make_exampleframe_premedian(self,t,z,video_name,frame_range,medpose):    
        """
        Make an example frame showing the detections of all of the networks in the ensemble toether, as well as the median pose. This is the case where you have pre-computed the median pose and can pass it to the function. 
        :param t: the frame index to make an example frame from.
        :param z: the length of trace history to show on the frame, in frame indicees
        :param video_name: name of a moviepy video. 
        :param frame_range: range of frames we are calculating over. 
        :param medpose: the median pose for the frame t. Should be of shape (xy,part)  
        """
        assert type(frame_range) == type(range(0,1))
        poses = self.get_poses(video_name)
        ensemble_pose = {}
        for key,p in poses.items():
            ensemble_pose[key] = p[frame_range,:,:]

        clip = self.get_video_clip(video_name,frame_range)    
        plt.imshow(clip.get_frame(t/clip.fps))
        for i in range(len(self.models)):
            for part in range(ensemble_pose["run{}".format(i)].shape[-1]):
                if part == 0:
                    plt.plot(*ensemble_pose["run{}".format(i)][t,:,part],
                            "o",
                            markersize = 4,
                            marker = markers[part],
                            linestyle = None,
                            color = colors[i],
                            label = "run{}".format(i),
                            alpha = 0.5)
                else:    
                    plt.plot(*ensemble_pose["run{}".format(i)][t,:,part],
                            "o",
                            markersize = 4,
                            marker = markers[part],
                            linestyle = None,
                            color = colors[i],
                            alpha = 0.5)
        for part in range(ensemble_pose["run{}".format(i)].shape[-1]):
            if part == 0:
                plt.plot(*medpose[:,part],
                        "x",
                        linestyle = 'None',
                        markersize = 5,
                        marker = markers[part],
                        label = "median",
                        color = colors[-1])
            else:    
                plt.plot(*medpose[:,part],
                        "x",
                        linestyle = 'None',
                        markersize = 5,
                        marker = markers[part],
                        color = colors[-1])
        plt.axis("off")        
        plt.legend() 
        return plt.gcf()

    def make_exampleframe(self,t,z,video_name,frame_range):    
        """
        Make an example frame showing the detections of all of the networks in the ensemble toether, as well as the median pose. 
        :param t: the frame index to make an example frame from.
        :param z: the length of trace history to show on the frame, in frame indicees
        :param video_name: name of a moviepy video. 
        :param frame_range: range of frames we are calculating over. 
        """
        assert type(frame_range) == type(range(0,1))
        poses = self.get_poses(video_name)
        ensemble_pose = {}
        for key,p in poses.items():
            ensemble_pose[key] = p[frame_range,:,:]
        rawpose = self.get_mean_pose(video_name,range(t,t+2))    
        medpose = np.stack(rawpose,axis = 1) 

        clip = self.get_video_clip(video_name,frame_range)    
        plt.imshow(clip.get_frame(t/clip.fps))
        for i in range(len(self.models)):
            for part in range(ensemble_pose["run{}".format(i)].shape[-1]):
                if part == 0:
                    plt.plot(*ensemble_pose["run{}".format(i)][t,:,part],
                            "o",
                            markersize = 4,
                            marker = markers[part],
                            linestyle = "None",
                            color = colors[i],
                            label = "run{}".format(i),
                            alpha = 0.5)
                else:    
                    plt.plot(*ensemble_pose["run{}".format(i)][t,:,part],
                            "o",
                            markersize = 4,
                            marker = markers[part],
                            linestyle = "None",
                            color = colors[i],
                            alpha = 0.5)
        for part in range(ensemble_pose["run{}".format(i)].shape[-1]):
            if part == 0:
                plt.plot(*medpose[0,:,part],
                        "x",
                        linestyle = 'None',
                        markersize = 5,
                        marker = markers[part],
                        label = "consensus",
                        color = colors[-1])
            else:    
                plt.plot(*medpose[0,:,part],
                        "x",
                        linestyle = 'None',
                        markersize = 5,
                        marker = markers[part],
                        color = colors[-1])
        plt.axis("off")        
        plt.legend(numpoints =1) 
        return plt.gcf()
        
    def compare_groundtruth(self,videoname,groundtruthpath,partperm = None,indices = None,parts = None):    
        """Like the TrainedModel method of the same name, get the groundtruth trace and compare to each member of the ensemble + the median.   
        :param labeled_video: Name of the video that data is provided for. 
        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        :param indices: a numpy array of indices to run this comparison for. 
        :param parts: the parts we should include when computing the groundtruth. Indexed via the parts in the ensemble pose detections, not the groundtruth. Must be given as a 1d numpy array.  
        """
        rmses = {}
        for modelname,model in self.models.items():
            rmses[modelname] = model.compare_groundtruth(videoname,groundtruthpath,partperm,indices = indices,parts = parts)
        ## Use the last model to calculate this (no dependencies on that model's params that aren't general to the ensemble. )    
        gt = model.get_groundtruth(groundtruthpath,partperm)  
        gtlength = len(gt)+1

        ## Finally get the mean pose:     

        rawmedpose = self.get_mean_pose(videoname,range(gtlength))    
        medpose = np.stack(rawmedpose,axis = 1) 

        if parts is None:
            parts = np.array(np.arange(gt.shape[-1]))
        else:    
            assert type(parts) == np.ndarray
            assert len(parts.shape) == 1

        if indices is None:
            medrmse = model.marker_epsilon_distance(medpose[:len(gt),:,parts],gt)        
        else:    
            medpose_reshaped = medpose[indices[:,None,None],np.array([0,1])[:,None],parts] ## a pain to reshape for missing parts...
            medrmse = model.marker_epsilon_distance(medpose_reshaped,gt[indices,:,:])        
            print(medpose[indices,:,:].shape)
        rmses["median"] = medrmse
        return rmses

class TrainedModel():
    """Trained DGP model. Initialized by passing a model folder. Once initialized can be queried for trace data videos, etc. 

    :param projectfolder: path to project this model is initialized from. 
    :param ext: file extension for videos (e.g. avi, mp4)
    :param memory: location of joblib cache. By default, is the location declared at the top of this module. If set to None, will not save to the cache.  

    """
    def __init__(self,projectfolder,ext,memory = memory):
        """Initializes with a project folder, and associates the relevant video files.

        """
        self.project_dir = Path(projectfolder)
        self.ext = ext
        self.load_videos()
        self.change_project_path()
        self.memory = memory

    def load_videos(self):    
        """Reloads the list of videos and label files from the file system. 

        """
        self.pred_video_files = os.listdir(self.project_dir/"videos_pred")
        self.pred_videos = [video_file_ for video_file_ in self.pred_video_files if self.ext in video_file_]
        self.label_files = [video_file_ for video_file_ in self.pred_video_files if '.csv' in video_file_]

    def change_project_path(self):    
        yamlpath = os.path.join(self.project_dir,"config.yaml")
        with open(yamlpath,"r") as f:
            output = yaml.safe_load(f)
        video_sets_trunc = {}    
        for k,v in output["video_sets"].items():
            trunc_path = k.split(os.path.join(output["project_path"],""))[-1]
            video_sets_trunc[os.path.join(self.project_dir,trunc_path)] = v
        output["project_path"] = str(self.project_dir)
        output["video_sets"] = video_sets_trunc
        with open(yamlpath,"w") as f:
            yaml.dump(output,f)

    def predict(self,orig_video_path,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Performs prediction with this model onto the given video. 

        :param orig_video_path: full path to the video file. Video will also be copied into the model's videos directory. 
        :param snapshot: the model snapshot to use. defaults to 'snapshot-step2-final--0'
        :param shuffle: if model was used for xvalidation, gives the train/test split relevant 
        """
        video_name = os.path.basename(orig_video_path)

        video_dir = os.path.join(self.project_dir,"videos_pred")
        rawviddir = os.path.join(self.project_dir,"videos")
        #video_path = os.path.join(video_dir,video_name)

        rawvidpath = os.path.join(rawviddir,video_name)
        shutil.copyfile(orig_video_path,rawvidpath)
        

        print('Collecting markers from snapshot:')
        print(snapshot)
        print('\n')

        #full_video_clip = self.get_video_clip(video_path,None)
        #fps = full_video_clip.fps
        snapshot_path, cfg_yaml = get_snapshot_path(snapshot, self.project_dir, shuffle=shuffle)
        print(cfg_yaml)
        plot_dgp(str(orig_video_path),
                str(video_dir),
                proj_cfg_file = str(cfg_yaml),
                dgp_model_file = str(snapshot_path),
                shuffle = shuffle)

        self.load_videos()

    def get_poses_raw(self,video_name):    
        """Gets the pose for a video that has been predicted on. 
        
        :param video_name:
        :returns: the output of deepgraphpose.models.eval.load_pose_from_dlc_to_dict. This dictionary has keys "x","y","likelihoods", and 
        """
        assert video_name in self.pred_video_files, "video must have been predicted on."
        csv_name = os.path.splitext(video_name)[0]+".csv"
        assert csv_name in self.label_files, "label file must exist."  
        path = os.path.join(self.project_dir,"videos_pred",csv_name)
        labels = load_pose_from_dlc_to_dict(path)
        return labels

    def get_poses_array(self,video_name):
        """Processes the pose into an array of shape (time,xy,body part) 

        """
        labels = self.get_poses_raw(video_name)
        xr,yr = labels["x"],labels["y"]
        labelarray = np.stack((xr,yr),axis = 1)
        return labelarray

    def get_video_clip(self,video_name,frame_range = None):
        """Get a subclip of a video. Note strange behavior where using iter_frames: 

        """
        assert video_name in self.pred_video_files, "video must have been predicted on."
        video_path = os.path.join(self.project_dir,"videos_pred",video_name)
        clip = get_video_clip(video_path,frame_range = frame_range)
        return clip

    def get_poses_and_heatmap(self,video_name,framenb,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the pose and heatmap information of a predicted video for n_frames consecutive frames, starting from the first in a subclip determined by frame_range. Originally given as get_body in plot_cmap5. Note strange behavior with iter frames: iter_frames on a clip with one frame does not give that first frame. 
        :param video_name: name of the video. 
        :param framenb: frame to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        """
        print('Collecting markers from snapshot:')
        print(snapshot)
        print('\n')

        full_video_clip = self.get_video_clip(video_name,None)
        fps = full_video_clip.fps
        frame = full_video_clip.get_frame(framenb/full_video_clip.fps)

        snapshot_path, cfg_yaml = get_snapshot_path(snapshot, self.project_dir, shuffle=shuffle)
        cfg = auxiliaryfunctions.read_config(cfg_yaml)

        proj_cfg_file = str(cfg_yaml)
        dgp_model_file = str(snapshot_path)
        # %% estimate_pose
        # load dlc project config file
        print('loading dlc project config...')
        with open(proj_cfg_file, 'r') as stream:
            proj_config = yaml.safe_load(stream)
        proj_config['video_path'] = None
        proj_config["project_path"] = self.project_dir ## not running postprocessing in same place as inference, must change. 

        dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
        print('done')
        # %%
        try:
            dlc_cfg.net_type = 'resnet_50'
            sess, mu_n, softmax_tensor, scmap, locref, inputs = setup_dgp_eval_graph(dlc_cfg,
                                                                                     dgp_model_file)
        except:
            dlc_cfg.net_type = 'resnet_101'
            sess, mu_n, softmax_tensor, scmap, locref, inputs = setup_dgp_eval_graph(dlc_cfg,
                                                                                     dgp_model_file)
        # %%
        nj = dlc_cfg.num_joints
        bodyparts = cfg['bodyparts']
        # %%
        nx, ny = full_video_clip.size
        nx_out, ny_out = int((nx - dlc_cfg.stride/2)/dlc_cfg.stride  + 5), int((ny - dlc_cfg.stride/2)/dlc_cfg.stride  + 5)
        #%%
        markers = np.zeros((2, dlc_cfg.num_joints, 2))
        likes = np.zeros((2, dlc_cfg.num_joints))
        softmaxtensor = np.zeros((2, ny_out, nx_out, dlc_cfg.num_joints))
        ii = 0
        
        ff = img_as_ubyte(frame)
        mu_n_batch, pred_np1 = sess.run( [mu_n,  scmap], feed_dict={inputs: ff[None, :, :, :]})
        nx_out_true, ny_out_true, _ = pred_np1[0].shape
        markers[ii] = mu_n_batch[0]
        num_frames = mu_n_batch.shape[0]  # nb #+ 1
        sigmoid_pred_np = np.exp(pred_np1) / (np.exp(pred_np1) + 1)        
        #softmaxtensor[ii][:nx_out_true,:ny_out_true] = sigmoid_pred_np[0]
        softmaxtensor[ii][:nx_out_true,:ny_out_true] = pred_np1[0]
        mu_likelihoods = np.zeros((num_frames, nj, 2)).astype('int')
        likelihoods = np.zeros((num_frames, nj))
        offset_mu_jj = 0
        for ff_idx in range(num_frames):
            for jj_idx in range(nj):
                # continuous so pick max in
                mu_jj = mu_n_batch[ff_idx, jj_idx]
                ends_floor = np.floor(mu_jj).astype('int') - offset_mu_jj
                ends_ceil = np.ceil(mu_jj).astype('int') + 1 + offset_mu_jj
                sigmoid_pred_np_jj = sigmoid_pred_np[ff_idx, :, :, jj_idx]
                spred_centered = sigmoid_pred_np_jj[ ends_floor[0]: ends_ceil[0],
                                 ends_floor[1]: ends_ceil[1]]
                mu_likelihoods[ff_idx, jj_idx] = np.unravel_index(
                    np.argmax(spred_centered), spred_centered.shape)
                mu_likelihoods[ff_idx, jj_idx] += [ends_floor[0], ends_floor[1]]
                likelihoods[ff_idx, jj_idx] = sigmoid_pred_np_jj[
                    int(mu_likelihoods[ff_idx, jj_idx][0]), int(mu_likelihoods[ff_idx, jj_idx][1])]
        likes[ii] = likelihoods[0]

        likes = likes[:ii+1]
        markers = markers[:ii+1]
        softmaxtensor = softmaxtensor[:ii+1,:nx_out_true,:ny_out_true,:]
        pbar.close()
        sess.close()
        full_video_clip.close()
        print('Finished collecting markers')
        print('\n')
        xx = markers[:, :, 1] * dlc_cfg.stride + 0.5 * dlc_cfg.stride
        yy = markers[:, :, 0] * dlc_cfg.stride + 0.5 * dlc_cfg.stride

        return xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg

    def get_poses_and_heatmap_range(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the pose and heatmap information of a predicted video for n_frames consecutive frames, starting from the first in a subclip determined by frame_range. Originally given as get_body in plot_cmap5. Note strange behavior with iter frames: iter_frames on a clip with one frame does not give that first frame. 
        :param video_name: name of the video. 
        :param frame_range: range of frames to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        """
        assert type(frame_range) == type(range(0,2))
        print('Collecting markers from snapshot:')
        print(snapshot)
        print('\n')
        full_video_clip = self.get_video_clip(video_name,None)
        n_frames2 = min(frame_range[-1] + 1, np.ceil(full_video_clip.fps * full_video_clip.duration).astype('int'))
        offset=frame_range[0] 
        video_clip = self.get_video_clip(video_name,range(offset,n_frames2))
        fps = video_clip.fps
        snapshot_path, cfg_yaml = get_snapshot_path(snapshot, self.project_dir, shuffle=shuffle)
        cfg = auxiliaryfunctions.read_config(cfg_yaml)

        proj_cfg_file = str(cfg_yaml)
        dgp_model_file = str(snapshot_path)
        # %% estimate_pose
        # load dlc project config file
        print('loading dlc project config...')
        with open(proj_cfg_file, 'r') as stream:
            proj_config = yaml.safe_load(stream)
        proj_config['video_path'] = None
        proj_config["project_path"] = self.project_dir ## not running postprocessing in same place as inference, must change. 

        dlc_cfg = get_train_config(proj_config, shuffle=shuffle)
        print('done')
        # %%
        try:
            dlc_cfg.net_type = 'resnet_50'
            sess, mu_n, softmax_tensor, scmap, locref, inputs = setup_dgp_eval_graph(dlc_cfg,
                                                                                     dgp_model_file)
        except:
            dlc_cfg.net_type = 'resnet_101'
            sess, mu_n, softmax_tensor, scmap, locref, inputs = setup_dgp_eval_graph(dlc_cfg,
                                                                                     dgp_model_file)
        # %%
        nj = dlc_cfg.num_joints
        bodyparts = cfg['bodyparts']
        # %%
        nx, ny = video_clip.size
        nx_out, ny_out = int((nx - dlc_cfg.stride/2)/dlc_cfg.stride  + 5), int((ny - dlc_cfg.stride/2)/dlc_cfg.stride  + 5)
        #%%
        markers = np.zeros((n_frames2 + 1, dlc_cfg.num_joints, 2))
        likes = np.zeros((n_frames2 + 1, dlc_cfg.num_joints))
        softmaxtensor = np.zeros((n_frames2 + 1, ny_out, nx_out, dlc_cfg.num_joints))
        pbar = tqdm(total=n_frames2, desc='processing video frames')
        for ii, frame in enumerate(video_clip.iter_frames()):
            ff = img_as_ubyte(frame)
            mu_n_batch, pred_np1 = sess.run( [mu_n,  scmap], feed_dict={inputs: ff[None, :, :, :]})
            nx_out_true, ny_out_true, _ = pred_np1[0].shape
            markers[ii] = mu_n_batch[0]
            num_frames = mu_n_batch.shape[0]  # nb #+ 1
            sigmoid_pred_np = np.exp(pred_np1) / (np.exp(pred_np1) + 1)        
            #softmaxtensor[ii][:nx_out_true,:ny_out_true] = sigmoid_pred_np[0]
            softmaxtensor[ii][:nx_out_true,:ny_out_true] = pred_np1[0]
            mu_likelihoods = np.zeros((num_frames, nj, 2)).astype('int')
            likelihoods = np.zeros((num_frames, nj))
            offset_mu_jj = 0
            for ff_idx in range(num_frames):
                for jj_idx in range(nj):
                    # continuous so pick max in
                    mu_jj = mu_n_batch[ff_idx, jj_idx]
                    ends_floor = np.floor(mu_jj).astype('int') - offset_mu_jj
                    ends_ceil = np.ceil(mu_jj).astype('int') + 1 + offset_mu_jj
                    sigmoid_pred_np_jj = sigmoid_pred_np[ff_idx, :, :, jj_idx]
                    spred_centered = sigmoid_pred_np_jj[ ends_floor[0]: ends_ceil[0],
                                     ends_floor[1]: ends_ceil[1]]
                    mu_likelihoods[ff_idx, jj_idx] = np.unravel_index(
                        np.argmax(spred_centered), spred_centered.shape)
                    mu_likelihoods[ff_idx, jj_idx] += [ends_floor[0], ends_floor[1]]
                    likelihoods[ff_idx, jj_idx] = sigmoid_pred_np_jj[
                        int(mu_likelihoods[ff_idx, jj_idx][0]), int(mu_likelihoods[ff_idx, jj_idx][1])]
            likes[ii] = likelihoods[0]
            pbar.update(1)

        likes = likes[:ii+1]
        markers = markers[:ii+1]
        softmaxtensor = softmaxtensor[:ii+1,:nx_out_true,:ny_out_true,:]
        pbar.close()
        sess.close()
        video_clip.close()
        print('Finished collecting markers')
        print('\n')
        xx = markers[:, :, 1] * dlc_cfg.stride + 0.5 * dlc_cfg.stride
        yy = markers[:, :, 0] * dlc_cfg.stride + 0.5 * dlc_cfg.stride

        return xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg
        
    def get_poses_and_heatmap_cache(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the pose and heatmap information of a predicted video for n_frames consecutive frames, starting from the first in a subclip determined by frame_range. Originally given as get_body in plot_cmap5. Note strange behavior with iter frames: iter_frames on a clip with one frame does not give that first frame. IMPORTANT: give the name of the video in the folder videos_pred: we will then look for the corresponding video in the folder videos.
        :param video_name: name of the labeled video. 
        :param frame_range: range of frames to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        """
        assert type(frame_range) == type(range(0,2))
        print('Collecting markers from snapshot:')
        print(snapshot)
        print('\n')
        snapshot_path, cfg_yaml = get_snapshot_path(snapshot, self.project_dir, shuffle=shuffle)
        cfg = auxiliaryfunctions.read_config(cfg_yaml)

        proj_cfg_file = str(cfg_yaml)
        dgp_model_file = str(snapshot_path)
        # %% estimate_pose
        # load dlc project config file
        print('loading dlc project config...')
        with open(proj_cfg_file, 'r') as stream:
            proj_config = yaml.safe_load(stream)
        proj_config['video_path'] = None
        proj_config["project_path"] = self.project_dir ## not running postprocessing in same place as inference, must change. 

        ## Given the name of the labeled video in videos_pred, we will need to run the actual tracking on the unlabeled video in videos.  
        video,ext = os.path.splitext(video_name)
        nolabel_list = video.split("_labeled")[:-1]
        video_nolabel = "".join(nolabel_list)+ext
        video_path = os.path.join(self.project_dir,"videos",video_nolabel)
        assert os.path.exists(video_path),"the video {} must exist in videos folder".format(video_nolabel)
        if self.memory is not None:
            cached_heatmap_compute = self.memory.cache(_get_poses_and_heatmap)
            xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg = cached_heatmap_compute(video_path,
                    frame_range,
                    cfg,
                    proj_config,
                    shuffle,
                    dgp_model_file)
        else:
            xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg = _get_poses_and_heatmap(video_path,
                    frame_range,
                    cfg,
                    proj_config,
                    shuffle,
                    dgp_model_file)


        return xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg

    def get_scoremap(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1,return_cfg = False):
        """Get scoremap (unnormalized likelihoods) from the convnet output. 

        :param video_name: name of the labeled video. 
        :param frame_range: range of frames to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        :param return_cfg: default false- if true, returns a tuple that includes the dlc cfg with formatting info for the scoremap. 
        :returns: array of shape (len(frame_range),48,58,parts)
        """
        xr,yr,likes,nj,bodyparts,scmap,dlc_cfg = self.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)
        if return_cfg == False:
            return scmap    
        else:
            return scmap,dlc_cfg    

    def get_logistic(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1,return_cfg = False):
        """ Get standard logistic transformation of scoremap. 

        :param video_name: name of the labeled video. 
        :param frame_range: range of frames to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        :param return_cfg: default false- if true, returns a tuple that includes the dlc cfg with formatting info for the scoremap. 
        :returns: array of shape (len(frame_range),48,58,parts)
        """
        scmap = self.get_scoremap(video_name,frame_range,snapshot,shuffle,return_cfg = return_cfg)
        if return_cfg == False:
            logistic=  np.exp(scmap)/(np.exp(scmap)+1)
            return logistic
        else:
            scmap_proper,cfg = scmap
            logistic=  np.exp(scmap_proper)/(np.exp(scmap_proper)+1)
            return logistic,cfg

    def get_groundtruth(self,groundtruth_path,partperm = None):
        """Get groundtruth data. 

        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        :returns: groundtruth_permuted- an array of shape (frames,xy,len(partperm)/parts)
        """
        groundtruth = loadmat(groundtruth_path)["true_xy"]
        groundtruth_reordered = np.moveaxis(groundtruth,0,-1) ## move parts to the last axis. 
        
        if partperm is None:
            groundtruth_permuted = groundtruth_reordered
        else:
            assert len(partperm) == groundtruth_reordered.shape[-1]
            assert np.all(np.sort(partperm) == np.sort(range(len(partperm))))
            groundtruth_permuted = groundtruth_reordered[:,:,np.array(partperm)]
        return groundtruth_permuted

    def get_groundtruth_confidence(self,groundtruth_path,video_name,frame_range,partperm = None,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Get the confidence output (normalized between [0,1]) corresponding to the groundtruth detection. Right now, calculates this by rounding the groundtruth detection into heatmap coordinates- we could also accomplish this by 2D interpolation of the scoremap in the future. 

        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        :param video_name: name of the labeled video. 
        :param frame_range: range of frames to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        :returns: array of scores of shape (frames,parts) that gives the confidence at the groundtruth location in each frame for each part.  
        """
        ## First get the groundtruth: 
        groundtruth = self.get_groundtruth(groundtruth_path,partperm)
        ## Now get the logistic map: 
        logistic_scoremaps,cfg  = self.get_logistic(video_name,frame_range,snapshot = snapshot, shuffle = shuffle,return_cfg = True)
        ## Now reformat the groundtruth to match the stride of the outputs: 
        groundtruth_seq = groundtruth[frame_range[0]:frame_range[-1],:,:]
        groundtruth_scaled = (groundtruth_seq-0.5*cfg.stride)/(cfg.stride) ## scale the groundtruth down to the heatmaps. 
        gt_scaled_int = np.round(groundtruth_scaled).astype(int)
        ## index into this array. 
        ### 1. construct indexing arrays for first and last dimensions. These are like the meshgrids, but with matrix indexing: . 
        shape = np.shape(gt_scaled_int)
        framesgrid,partsgrid = np.meshgrid(np.arange(shape[0]),np.arange(shape[-1]),indexing = "ij")
        scores = logistic_scoremaps[framesgrid,gt_scaled_int[:,1,:],gt_scaled_int[:,0,:],partsgrid] ## gotta be careful about the indexing here! Not consistent between indexing into the matrix and plotting.  
        return scores

    def get_groundtruth_probability(self,groundtruth_path,video_name,frame_range,partperm = None,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Get the probability output (normalized between [0,1] AND normalized across the entire scoremap) corresponding to the groundtruth detection. Right now, calculates this by rounding the groundtruth detection into heatmap coordinates- we could also accomplish this by 2D interpolation of the scoremap in the future. 

        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        :param video_name: name of the labeled video. 
        :param frame_range: range of frames to predict on. 
        :param snapshot: the name of the training snapshot to apply this analysis to. 
        :returns: array of scores of shape (frames,parts) that gives the probability (normalized confidence) at the groundtruth location in each frame for each part.  
        """
        ## First get the groundtruth: 
        groundtruth = self.get_groundtruth(groundtruth_path,partperm)
        ## Now get the logistic map: 
        logistic_scoremaps,cfg  = self.get_logistic(video_name,frame_range,snapshot = snapshot, shuffle = shuffle,return_cfg = True) ## scoremaps have shape (frame_range,x,y,parts)
        logistic_mass = np.sum(logistic_scoremaps,axis = (1,2),keepdims=True)
        logistic_probs = logistic_scoremaps/logistic_mass
        ## Now reformat the groundtruth to match the stride of the outputs: 
        groundtruth_seq = groundtruth[frame_range[0]:frame_range[-1],:,:]
        groundtruth_scaled = (groundtruth_seq-0.5*cfg.stride)/(cfg.stride) ## scale the groundtruth down to the heatmaps. 
        gt_scaled_int = np.round(groundtruth_scaled).astype(int)
        ## index into this array. 
        ### 1. construct indexing arrays for first and last dimensions. These are like the meshgrids, but with matrix indexing: . 
        shape = np.shape(gt_scaled_int)
        framesgrid,partsgrid = np.meshgrid(np.arange(shape[0]),np.arange(shape[-1]),indexing = "ij")
        scores = logistic_probs[framesgrid,gt_scaled_int[:,1,:],gt_scaled_int[:,0,:],partsgrid] ## gotta be careful about the indexing here! Not consistent between indexing into the matrix and plotting.  
        return scores
    def compare_groundtruth_pointwise(self,labeled_video,groundtruth_path,partperm = None,indices = None,parts=None):
        """Compare groundtruth data to detections pointwise and get framewise differences. Assumes we want to take groundtruth comparison for whole sequence unless indices are explicitly provided. 

        :param labeled_video: Name of the video that data is provided for. 
        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        :param indices: The frame indices we should include when computing comparison to groundtruth.  
        :param parts: the parts we should include when computing the groundtruth. Indexed via the parts in the ensemble pose detections, not the groundtruth. Must be given as a 1d numpy array.  
        """
        video_clip = self.get_video_clip(labeled_video,None)
        poses = self.get_poses_array(labeled_video)
        groundtruth = self.get_groundtruth(groundtruth_path,partperm)
        if parts is None:
            parts = np.array(np.arange(groundtruth.shape[-1]))
        else:    
            assert type(parts) == np.ndarray
            assert len(parts.shape) == 1
        ## Index into the array appropriately    
        if indices is None: 
            pose_eval = poses[:len(groundtruth),:,parts]
            groundtruth_eval = groundtruth
        else:    
            pose_eval = poses[indices[:,None,None],np.array([0,1])[:,None],parts]
            groundtruth_eval = groundtruth[indices,:,:]
        return groundtruth_eval-pose_eval    

    def compare_groundtruth(self,labeled_video,groundtruth_path,partperm = None,indices = None,parts=None):
        """Compare to groundtruth detected data and get rmse. Assumes that we have groundtruth for the whole sequence unless indices are explicitly provided. 

        :param labeled_video: Name of the video that data is provided for. 
        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        :param parts: the parts we should include when computing the groundtruth. Assumed that these parts are EXCLUDED from the given groundtruth. Indexed via the parts in the ensemble pose detections, not the groundtruth. Must be given as a 1d numpy array.  
        """
        video_clip = self.get_video_clip(labeled_video,None)
        poses = self.get_poses_array(labeled_video)
        groundtruth = self.get_groundtruth(groundtruth_path,partperm)
        if parts is None:
            parts = np.array(np.arange(groundtruth.shape[-1]))
        else:    
            assert type(parts) == np.ndarray
            assert len(parts.shape) == 1


        ## calculate rmse: 
        #rmse = np.sqrt(np.mean((poses[:len(groundtruth),:,:] - groundtruth)**2))
        if indices is None:
            rmse = self.marker_epsilon_distance(poses[:len(groundtruth),:,parts],groundtruth)
        else:    
            assert type(indices) == np.ndarray
            poses_reshaped = poses[indices[:,None,None],np.array([0,1])[:,None],parts] ## a pain to reshape for missing parts...
            rmse = self.marker_epsilon_distance(poses_reshaped,groundtruth[indices,:,:])

        return rmse

    def marker_epsilon_distance(self,pose,gt,epsilon=0):
        """Kelly's rmse code (for TxD) adapted to TxCxD (C = coordinate, D = part). Check this with kelly code again. 

        :param pose: the auto detected pose that we are working with. Of shape (time,xy,part)
        :param groundtruth: the groundtruth pose that we are working with. Of shape (time,xy,part)
        """
        dcoord = np.sum((pose-gt)**2,axis = 1) # now TxD, the rest of the code is the same 
        indiv_dist = np.sqrt(dcoord)
        ## epsilon radius for distance: 
        indiv_dist_epsilon=indiv_dist < epsilon
        indiv_dist[indiv_dist_epsilon] = 0
        indiv_dist2 = indiv_dist**2
        indiv_dist2 = indiv_dist2.sum(-1) # squared error per frame. 
        num_points = np.prod(indiv_dist2.shape) ## number of timepoints
        dist = np.sqrt(indiv_dist2.sum()/num_points) ## root mean square error. 
        #dist = np.sqrt(indiv_dist2)
        return dist  # T

