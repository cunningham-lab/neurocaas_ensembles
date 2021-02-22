## Code to handle trained dgp models.  
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
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg',warn=False, force=True)
from matplotlib import pyplot as plt
from matplotlib import animation
from moviepy.editor import VideoFileClip,VideoClip
from joblib import Memory
if os.getenv("HOME") == "/Users/taigaabe": 
    location = '/Volumes/TOSHIBA EXT STO/cache'
else:    
    location = os.path.join(os.getenv("HOME"),"cache")
memory = Memory(location, verbose=0)

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

    """
    def __init__(self,topdir,modeldirs,ext):
        self.topdir = topdir
        self.modeldirs = modeldirs
        self.modelpaths = [os.path.abspath(os.path.join(self.topdir,self.modeldirs[i])) for i in range(len(self.modeldirs))]
        for mp in self.modelpaths:
            assert os.path.exists(mp) , f"dir {mp} does not exist."
        self.models = {mi:TrainedModel(mp,ext) for mi,mp in enumerate(self.modelpaths)}
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

    def get_scoremaps(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        for model in self.models:
            xx,yy,likes,nj,bodyparts,softmaxtensor,dlc_cfg = model.get_poses_and_heatmap_info(video_name,frame_range,snapshot,shuffle)

    def get_mean_pose(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the scoremaps across the ensemble for this frame range of this video at this snapshot, and calculates the mean pose from it.  
        NOTE: passing frame_range(0,2) will give 1 frame, not 2 as you would expect.  
        TODO: Write test for this.  
        :param video_name:
        :param frame_range:
        :param snapshot:
        :param shuffle:

        """
        softmaxtensors = []
        for i in range(len(self.models)):
            model = self.models[i]
            xr,yr,likes,nj,bodyparts,softmaxtensor,dlc_cfg = model.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)
            softmaxtensors.append(softmaxtensor)    
        ref_ = np.mean(softmaxtensors,0)    
        ref_x = np.empty_like(xr)
        ref_y = np.empty_like(yr)
        len_range = len(frame_range)-1 ## just index relative to the subclip. Strangely moviepy returns len(framerange) -1 frames...
        for nt0 in range(len_range):
            for njj0 in range(nj):
                ref_y[nt0,njj0], ref_x[nt0,njj0] = np.unravel_index(np.argmax(ref_[nt0,:,:,njj0]), ref_.shape[1:3])
        ref_x = ref_x* dlc_cfg.stride + 0.5 * dlc_cfg.stride
        ref_y = ref_y* dlc_cfg.stride + 0.5 * dlc_cfg.stride
                
        return ref_x,ref_y

    def get_median_pose(self,video_name,frame_range,snapshot = "snapshot-step2-final--0",shuffle = 1):
        """Gets the scoremaps across the ensemble for this frame range of this video at this snapshot, and calculates the median pose from it.  
        NOTE: passing frame_range(0,2) will give 1 frame, not 2 as you would expect. 
        :param video_name:
        :param frame_range:
        :param snapshot:
        :param shuffle:

        """
        softmaxtensors = []
        for i in range(len(self.models)):
            model = self.models[i]
            xr,yr,likes,nj,bodyparts,softmaxtensor,dlc_cfg = model.get_poses_and_heatmap_cache(video_name,frame_range,snapshot,shuffle)
            softmaxtensors.append(softmaxtensor)    
        ref_ = np.median(softmaxtensors,0)    
        ref_x = np.empty_like(xr)
        ref_y = np.empty_like(yr)
        len_range = len(frame_range)-1 ## just index relative to the subclip. Strangely moviepy returns len(framerange) -1 frames...
        for nt0 in range(len_range):
            for njj0 in range(nj):
                ref_y[nt0,njj0], ref_x[nt0,njj0] = np.unravel_index(np.argmax(ref_[nt0,:,:,njj0]), ref_.shape[1:3])
        ref_x = ref_x* dlc_cfg.stride + 0.5 * dlc_cfg.stride
        ref_y = ref_y* dlc_cfg.stride + 0.5 * dlc_cfg.stride
                
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
        rawpose = self.get_median_pose(video_name,range(t,t+2))    
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
                plt.plot(*medpose[0,:,part],
                        "x",
                        linestyle = 'None',
                        markersize = 5,
                        marker = markers[part],
                        label = "median",
                        color = colors[-1])
            else:    
                plt.plot(*medpose[0,:,part],
                        "x",
                        linestyle = 'None',
                        markersize = 5,
                        marker = markers[part],
                        color = colors[-1])
        plt.axis("off")        
        plt.legend() 
        return plt.gcf()
        
    def compare_groundtruth(self,videoname,groundtruthpath,partperm = None):    
        """Like the TrainedModel method of the same name, get the groundtruth trace and compare to each member of the ensemble + the median.   
        :param labeled_video: Name of the video that data is provided for. 
        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        """
        rmses = {}
        for modelname,model in self.models.items():
            rmses[modelname] = model.compare_groundtruth(videoname,groundtruthpath,partperm)
        gt = model.get_groundtruth(groundtruthpath,partperm)  
        gtlength = len(gt)+2

        ## Finally get the median pose:     
        rawmedpose = self.get_median_pose(videoname,range(gtlength))    
        medpose = np.stack(rawmedpose,axis = 1) 
        medrmse = np.sqrt(np.mean((medpose[:len(gt),:,:] - gt)**2))
        rmses["median"] = medrmse
        return rmses

class TrainedModel():
    """Trained DGP model. Initialized by passing a model folder. Once initialized can be queried for trace data videos, etc. 

    :param projectfolder: path to project this model is initialized from. 
    :param ext: file extension for videos (e.g. avi, mp4)

    """
    def __init__(self,projectfolder,ext):
        """Initializes with a project folder, and associates the relevant video files.

        """
        self.project_dir = Path(projectfolder)
        self.ext = ext
        self.pred_video_files = os.listdir(self.project_dir/"videos_pred")
        self.pred_videos = [video_file_ for video_file_ in self.pred_video_files if self.ext in video_file_]
        self.label_files = [video_file_ for video_file_ in self.pred_video_files if '.csv' in video_file_]

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
        """Gets the pose and heatmap information of a predicted video for n_frames consecutive frames, starting from the first in a subclip determined by frame_range. Originally given as get_body in plot_cmap5. Note strange behavior with iter frames: iter_frames on a clip with one frame does not give that first frame. 
        :param video_name: name of the video. 
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

        #vars: video_name,frame_range,proj_config,shuffle,dgp_model_file,  
        #def _get_poses_and_heatmap(video_path,frame_range,proj_config,shuffle,dgp_model_file):
        video_path = os.path.join(self.project_dir,"videos_pred",video_name)
        cached_heatmap_compute = memory.cache(_get_poses_and_heatmap)
        xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg = cached_heatmap_compute(video_path,
                frame_range,
                cfg,
                proj_config,
                shuffle,
                dgp_model_file)

        return xx, yy, likes, nj, bodyparts, softmaxtensor, dlc_cfg

    def get_groundtruth(self,groundtruth_path,partperm = None):
        """Get groundtruth data. 

        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
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
        
    def compare_groundtruth(self,labeled_video,groundtruth_path,partperm = None):
        """Compare to groundtruth detected data and get rmse. Assumes that we have groundtruth for the whole sequence.  

        :param labeled_video: Name of the video that data is provided for. 
        :param groundtruth_path: Path to groundtruth labeled data. Assumes that data at this path is a .mat file, with the entry data["true_xy"] a numpy array of shape (parts,time,xy) for the whole labeled video.   
        :param partperm: permute the ordering of parts in the groundtruth dataset to match the pose network output. 
        """
        video_clip = self.get_video_clip(labeled_video,None)
        poses = self.get_poses_array(labeled_video)
        groundtruth = self.get_groundtruth(groundtruth_path,partperm)

        ## calculate rmse: 
        rmse = np.sqrt(np.mean((poses[:len(groundtruth),:,:] - groundtruth)**2))

        return rmse

        

