## Test per-model code. 
import logging
import yaml
import numpy as np
from easydict import EasyDict
import pytest
import matplotlib.pyplot as plt
import dgp_ensembletools.models
import os

logging.basicConfig(level = logging.DEBUG)

loc = os.path.abspath(os.path.dirname(__file__))

class Test_Ensemble():
    def test_init(self):
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
    def test_get_video_clip(self):
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        clip = ens.get_video_clip("ibl1_labeled.mp4",(0,1000))
    def test_get_poses(self):
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        print(ens.get_poses("ibl1_labeled.mp4"))
    def test_make_exampleframe(self):    
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(4)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        fig = ens.make_exampleframe(75,4,"ibl1_labeled.mp4",range(0,1000))
        fig.savefig(os.path.join(loc,"test_outputs","test{}frame.png".format(75)))
    def test_get_scoremaps(self):
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(1)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        scmaps = ens.get_scoremaps("ibl1_labeled.mp4",range(0,4))
        logging.debug(scmaps)
    def test_get_logistic(self):   
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(1)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        softmax = ens.get_logistic("ibl1_labeled.mp4",range(0,4))
        for s in softmax:
            assert np.all(s<=1) and np.all(s>=0)
        logging.debug(softmax)
    def test_get_median_pose(self):
        topdir = os.path.join(loc,"../","data/ibl_data")
        modeldirs = ["{}".format(i+1) for i in range(1)]
        ens = dgp_ensembletools.models.Ensemble(topdir,modeldirs,ext="mp4")
        med = ens.get_median_pose("ibl1_labeled.mp4",range(0,4))
        print(med)
        logging.debug(med)


class Test_TrainedModel():
    def test_init(self):
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        assert "ibl1_labeled.mp4" in tm.pred_videos 
        assert "ibl1_labeled.csv" in tm.label_files 

    def test_change_project_path(self):
        relpath = "1/"
        modelpath = os.path.join(loc,"test_mats/",relpath,"")
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"test_mats/",relpath),ext= "mp4")
        with open(os.path.join(modelpath,"config.yaml"),"r") as f:
            oldoutput = yaml.safe_load(f) 
        tm.change_project_path()
        with open(os.path.join(modelpath,"config.yaml"),"r") as f:
            newoutput = yaml.safe_load(f) 
        print(oldoutput,newoutput)    
        assert modelpath.startswith(newoutput["project_path"])
        for k in newoutput["video_sets"]:
            assert k.startswith(modelpath)
        
    def test_predict(self):
        relpath = "1/"
        modelpath = os.path.join(loc,"test_mats/",relpath,"")
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"test_mats/",relpath),ext= "mp4")
        tm.predict(os.path.join(loc,"test_mats","testpred.mp4"))
        assert os.path.exists(os.path.join(modelpath,"videos_pred","testpred_labeled.mp4"))
        assert os.path.exists(os.path.join(modelpath,"videos","testpred.mp4"))

    def test_get_poses_raw(self):    
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        poses = tm.get_poses_raw("ibl1_labeled.mp4")
        for pk in poses.keys():
            assert pk in ["x","y","likelihoods"]
            assert type(poses[pk]) == np.ndarray

    def test_get_video_clip(self):
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        clip = tm.get_video_clip("ibl1_labeled.mp4",range(0,1000))
    def test_get_video_clip_fail(self):
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        with pytest.raises(AssertionError):
            clip = tm.get_video_clip("ibl1_labeled.mp4",(0,1000))
    def test_get_poses_array(self):
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        labelarray = tm.get_poses_array("ibl1_labeled.mp4")
        assert type(labelarray) == np.ndarray
        assert labelarray.shape[1:] == (2,4)

    @pytest.mark.parametrize("video",["ibl1.mp4","ibl1_labeled.mp4"])    
    def test_get_poses_and_heatmap_range(self,video):    
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        if video == "ibl1.mp4":
            with pytest.raises(AssertionError):
                all_outs = tm.get_poses_and_heatmap_range(video,frame_range = range(0,2))
        else:        
            all_outs = tm.get_poses_and_heatmap_range(video,frame_range = range(0,2))


    @pytest.mark.parametrize("video",["ibl1.mp4","ibl1_labeled.mp4"])    
    def test_get_poses_and_heatmap_cache(self,video):    
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        if video == "ibl1.mp4":
            with pytest.raises(AssertionError):
                all_outs = tm.get_poses_and_heatmap_cache(video,frame_range = range(0,2))
        else:
            all_outs = tm.get_poses_and_heatmap_cache(video,frame_range = range(0,2))


    @pytest.mark.xfail    
    def test_get_poses_and_heatmap(self):    
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        all_outs = tm.get_poses_and_heatmap("ibl1.mp4",framenb = 0)

    def test_get_scoremap(self):
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        scmap = tm.get_scoremap("ibl1_labeled.mp4",range(0,4))
        assert scmap.shape == (3,48,58,4) 
        scmap = tm.get_scoremap("ibl1_labeled.mp4",range(0,4),return_cfg = True)
        assert scmap[0].shape == (3,48,58,4) 
        assert type(scmap[1]) == EasyDict 
        logging.debug(scmap)

    def test_get_logistic(self):   
        relpath = "data/ibl_data/1/"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        softmax = tm.get_logistic("ibl1_labeled.mp4",range(0,4))
        assert np.all(softmax<=1) and np.all(softmax>=0)
        assert softmax.shape == (3,48,58,4)
        softmax = tm.get_logistic("ibl1_labeled.mp4",range(0,4),return_cfg = True)
        assert np.all(softmax[0]<=1) and np.all(softmax[0]>=0)
        assert softmax[0].shape == (3,48,58,4)
        assert type(softmax[1]) ==EasyDict 
        
    def test_get_groundtruth(self):
        relpath = "data/ibl_data/1/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"
        t = 305 

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        clip = tm.get_video_clip(videopath)
        out = tm.get_groundtruth(datapath)
        
        poses = tm.get_poses_array(videopath)
        frame = clip.get_frame(t/clip.fps)
        plt.imshow(frame)
        plt.plot(*poses[t,:,:],label = "auto",color = "blue")
        plt.plot(*out[t,:,:],label = "gt",color="orange")
        plt.plot(*poses[t,:,0],"o",markersize = 5,color = "blue")
        plt.plot(*out[t,:,0],"o",markersize = 5,color = "orange")
        plt.legend()
        plt.savefig(os.path.join(loc,"test_outputs",f"test_get_groundtruth_frame{t}.png"))

    def test_get_groundtruth_confidence(self):    
        relpath = "data/ibl_data/1/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        gt = tm.get_groundtruth(datapath,partperm = [1,3,0,2])
        sc,cfg  = tm.get_logistic(videopath,range(0,3),return_cfg = True)
        gt_0 = gt[0,:,0] ## xy for first part in first frame
        gt_0_scaled = (gt_0-0.5*cfg.stride)/(cfg.stride)
        gt_1_1 = gt[1,:,1] ## xy for second part in second frame
        gt_1_1_scaled = (gt_1_1-0.5*cfg.stride)/(cfg.stride)
        coords_0 = np.round(gt_0_scaled).astype(int)
        coords_1_1 = np.round(gt_1_1_scaled).astype(int)
        scores = tm.get_groundtruth_confidence(datapath,videopath,range(0,4),partperm = [1,3,0,2])
        assert scores[0,0] == sc[0,coords_0[0],coords_0[1],0]
        assert scores[1,1] == sc[1,coords_1_1[0],coords_1_1[1],1]
        plt.imshow(sc[0,:,:,0])
        plt.plot(*gt_0_scaled,"x",label = "manual")
        plt.savefig(os.path.join(loc,"test_outputs","scaled_groundtruth.png"))

    def test_get_groundtruth_perm(self):
        relpath = "data/ibl_data/1/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"
        t = 305 

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        clip = tm.get_video_clip(videopath)
        out = tm.get_groundtruth(datapath,partperm = [1,3,0,2])
        
        poses = tm.get_poses_array(videopath)
        frame = clip.get_frame(t/clip.fps)
        plt.imshow(frame)
        plt.plot(*poses[t,:,:],label = "auto",color = "blue")
        plt.plot(*out[t,:,:],label = "gt",color="orange")
        plt.plot(*poses[t,:,0],"o",markersize = 5,color = "blue")
        plt.plot(*out[t,:,0],"o",markersize = 5,color = "orange")
        plt.legend()
        plt.savefig(os.path.join(loc,"test_outputs",f"test_get_groundtruth_frame{t}_perm.png"))

    def test_compare_groundtruth_pointwise(self):
        relpath = "data/ibl_data/1/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"
        t = 305 
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        out = tm.get_groundtruth(datapath,partperm = [1,3,0,2])
        poses = tm.get_poses_array(videopath)[:len(out)]
        diffarray = out-poses
        ## base case
        assert np.all(tm.compare_groundtruth_pointwise("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2]) == diffarray)
        ## indexing into frames
        frames = np.array([0,1,2])
        assert np.all(tm.compare_groundtruth_pointwise("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],indices = frames) == diffarray[frames,:,:])
        ## To test below we need a groundtruth with missing parts.
        ### indexing into parts
        #parts = np.array([3,2])
        #assert np.all(tm.compare_groundtruth_pointwise("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],parts = parts) == diffarray[:,np.array([0,1])[:,None],parts])
        ### indexing into frames and parts:
        #assert np.all(tm.compare_groundtruth_pointwise("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],indices = frames,parts = parts) == diffarray[frames,np.array([0,1])[:,None],parts])

    def test_compare_groundtruth(self):    
        relpath = "data/ibl_data/2/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        out = tm.compare_groundtruth("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2])
        assert out < 41

    def test_compare_groundtruth_parts_fail(self):    
        relpath = "data/ibl_data/2/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        with pytest.raises(AssertionError):
            out = tm.compare_groundtruth("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],parts = [0])

    def test_compare_groundtruth_parts(self):    
        relpath = "data/ibl_data/2/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")
        out = tm.compare_groundtruth("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],parts = np.array([0]))
        noneout = tm.compare_groundtruth("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],parts =None)
        fullout = tm.compare_groundtruth("ibl1_labeled.mp4",datapath,partperm = [1,3,0,2],parts = np.array([0,1,2,3]))

        assert out < 56 
        assert noneout == fullout

    def test_marker_epsilon_distance(self):    
        relpath = "data/ibl_data/2/"
        datapath = "/Users/taigaabe/Downloads/ibl1_true_xy_all_918pm.mat"
        videopath = "ibl1_labeled.mp4"

        tm = dgp_ensembletools.models.TrainedModel(os.path.join(loc,"../",relpath),ext= "mp4")

        gt = tm.get_groundtruth(datapath,partperm = [1,3,0,2])
        poses = tm.get_poses_array(videopath)

        output = tm.marker_epsilon_distance(poses[:-1],gt)
        


